import pvporcupine, pyaudio, struct, whisper, wave, os, tempfile, threading, time, numpy as np
import ollama, pyttsx3, queue
from pathlib import Path
from typing import Optional
from collections import deque

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIG
# ════════════════════════════════════════════════════════════════════════════════════════════════════
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
WAKE_WORD_PATH = "Hey-Arthur_en_windows_v3_0_0.ppn"
MIC_INDEX, SAMPLE_RATE, CHUNK = 1, 16000, 512
SILENCE_THRESH, SILENCE_TIME, MIN_SPEECH = 500, 1.5, 0.3
OLLAMA_MODEL, MAX_TOKENS, TEMP = "llama3.2:3b", 150, 0.7
VOICE_RATE = 180

USER_INFO = """Name: Ronan
Age: 13
Location: Noels Pond, Newfoundland, Canada

Personality:
Smart-ass, adventurous, sarcastic, and funny. Dark sense of humor, usually jokes around and doesn't take things too seriously. Likes when Arthur talks like a friend — casual, witty, and not robotic.

Interests:
Loves gaming (Grounded 2, Roblox, Trailmakers) and tech. Excited for Subnautica 2 and the next Grounded update. Watches mostly gaming and tech content on YouTube.

Media & Fandoms:
Favorite shows are Stranger Things and Dark ("that cool German show"). Doesn't really have a favorite character but enjoys mysterious or sci-fi vibes.

School & Learning:
Enjoys gym, math, tech, and ELA. Tolerates social studies, French, and religion. Thinks health class is boring.

Sports & Activities:
Does Taekwondo during the school season, swims all year, and goes skiing in the winter when there's snow.

Setup:
Ryzen 5 5600X (6 core), 16GB RAM, 970 EVO Plus 1TB SSD, RX 6600 GPU, dual monitors, mouse, keyboard, and a Soomfon stream controller.

Pets:
• Sophie — 3 years old (Oct 2025), cute but energetic and not quite trained.
• Shadow — 3 months old (Oct 2025), mischievous but well-trained for a pup.
• Kenzy — 11 years old (Oct 2025), lazy and loves belly rubs.
• Boots — friendly cat, not shy.
• Bianca — shy cat, keeps to herself.

Friends & Family:
Has two close friends (Alex and Matheo, both male). Two siblings (Isaac — brother, Freya — sister). Calls parents simply Mom and Dad.
"""

CUSTOM_RESPONSES = {
    "brother iq": "Your brother's IQ is lower than a rock. Just kidding!",
    "brothers iq": "Your brother's IQ is lower than a rock. Just kidding!"
}

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# 🤖 ASSISTANT
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class VoiceAssistant:
    def __init__(self):
        self.history = deque(maxlen=10)
        self.wake_queue = queue.Queue()
        self.interrupt = threading.Event()
        self.speaking = False
        self.audio_lock = threading.Lock()
        
        # Check Ollama
        try:
            ollama.list()
            print(f"✅ Ollama: {OLLAMA_MODEL}")
        except:
            print("⚠️ Start Ollama with: ollama serve")
            exit(1)
        
        # Porcupine
        self.porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=[WAKE_WORD_PATH]
        )
        
        # Whisper
        print("🔄 Loading Whisper...")
        self.whisper = whisper.load_model("tiny")
        
        # Audio
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
            input_device_index=MIC_INDEX
        )
        
        print("✅ Arthur ready!")
    
    def monitor_wake_word(self):
        """Background wake word detection"""
        while True:
            try:
                with self.audio_lock:
                    if self.stream.is_active():
                        pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                    else:
                        time.sleep(0.1)
                        continue
                
                if self.porcupine.process(struct.unpack_from("h" * self.porcupine.frame_length, pcm)) >= 0:
                    if self.speaking:
                        self.interrupt.set()
                        print("\n⚠️ INTERRUPT!")
                    else:
                        self.wake_queue.put(time.time())
                        print("\n🟢 Wake word!")
            except Exception as e:
                print(f"⚠️ {e}")
                time.sleep(0.1)
    
    def record(self) -> Optional[str]:
        """Record audio until silence"""
        print("🎤 Listening...")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        
        try:
            with self.audio_lock:
                self.stream.stop_stream()
            
            rec_stream = self.pa.open(
                format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK, input_device_index=MIC_INDEX
            )
            
            # Clear buffer
            for _ in range(5):
                rec_stream.read(CHUNK, exception_on_overflow=False)
            time.sleep(0.2)
            
            frames, silence_cnt, speech_cnt = [], 0, 0
            max_silence = int(SILENCE_TIME * SAMPLE_RATE / CHUNK)
            min_speech = int(MIN_SPEECH * SAMPLE_RATE / CHUNK)
            speaking = False
            
            for _ in range(int(SAMPLE_RATE / CHUNK * 30)):
                data = rec_stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16).astype(np.float32))))
                
                if rms > SILENCE_THRESH:
                    speech_cnt += 1
                    silence_cnt = 0
                    if not speaking and speech_cnt >= 2:
                        speaking = True
                        print("   🗣️ Speaking...")
                else:
                    silence_cnt += 1
                
                if speaking and speech_cnt >= min_speech and silence_cnt > max_silence:
                    print("   ⏹️ Done")
                    break
            
            rec_stream.stop_stream()
            rec_stream.close()
            
            with self.audio_lock:
                self.stream.start_stream()
            
            if frames and len(frames) >= min_speech:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b"".join(frames))
                
                if os.path.getsize(tmp.name) > 1000:
                    return tmp.name
        except Exception as e:
            print(f"❌ Record error: {e}")
            with self.audio_lock:
                self.stream.start_stream()
        
        return None
    
    def transcribe(self, file: str) -> Optional[str]:
        """Convert speech to text"""
        if not file or not Path(file).exists():
            return None
        
        try:
            result = self.whisper.transcribe(file, language="en", fp16=False, temperature=0.0)
            text = result["text"].strip().replace("[BLANK_AUDIO]", "").strip()
            if text and len(text) > 2:
                return text
        except Exception as e:
            print(f"❌ Transcribe error: {e}")
        return None
    
    def ask(self, prompt: str) -> str:
        """Get AI response"""
        p = prompt.lower().replace('.', '').replace(',', '')
        
        # Custom responses
        for trigger, resp in CUSTOM_RESPONSES.items():
            if trigger in p:
                return resp
        
        try:
            print("🤔 Thinking...")
            
            msgs = [{
                "role": "system",
                "content": f"You are Arthur, Ronan's AI. Be casual, witty, concise (1-2 sentences).\n\n{USER_INFO}"
            }]
            
            for entry in list(self.history)[-3:]:
                msgs.append({"role": "user", "content": entry["user"]})
                msgs.append({"role": "assistant", "content": entry["assistant"]})
            
            msgs.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=msgs,
                options={"temperature": TEMP, "num_predict": MAX_TOKENS, "num_ctx": 2048}
            )
            
            reply = response['message']['content'].strip()
            self.history.append({"user": prompt, "assistant": reply})
            return reply
        except Exception as e:
            print(f"❌ AI error: {e}")
            return "Something went wrong. Is Ollama running?"
    
    def speak(self, text: str):
        """Text to speech - Fixed for Windows"""
        print(f"💬 Arthur: {text}")
        self.speaking = True
        self.interrupt.clear()
        
        try:
            # Create fresh TTS engine to avoid state issues on Windows
            engine = pyttsx3.init()
            engine.setProperty('rate', VOICE_RATE)
            engine.setProperty('volume', 1.0)
            
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            
            # Check for interrupt
            if self.interrupt.is_set():
                print("   ⚠️ Interrupted!")
                engine.stop()
                del engine
                return
            
            # Speak the full text
            engine.say(text)
            engine.runAndWait()
            
            # Clean up engine
            del engine
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
        finally:
            self.speaking = False
    
    def run(self):
        """Main loop"""
        threading.Thread(target=self.monitor_wake_word, daemon=True).start()
        print("🎙️ Say 'Hey Arthur' to begin.\n")
        
        try:
            while True:
                try:
                    self.wake_queue.get(timeout=0.1)
                    
                    # Clear queue
                    while not self.wake_queue.empty():
                        self.wake_queue.get_nowait()
                    
                    if self.speaking:
                        self.interrupt.set()
                        time.sleep(0.3)
                    
                    self.interrupt.clear()
                    
                    # Process
                    audio = self.record()
                    if audio:
                        cmd = self.transcribe(audio)
                        if cmd and len(cmd.strip()) > 2:
                            print(f"🗣️ You: {cmd}")
                            self.speak(self.ask(cmd))
                            try:
                                os.remove(audio)
                            except:
                                pass
                        else:
                            self.speak("I didn't catch that.")
                    else:
                        self.speak("Recording issue. Try again.")
                    
                    print(f"\n🎙️ Ready... ({len(self.history)} in memory)")
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        self.porcupine.delete()
        print("👋 Goodbye!")

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# 🚀 START
# ════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    VoiceAssistant().run()