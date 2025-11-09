import pvporcupine, pyaudio, struct, whisper, wave, os, tempfile, threading, time, numpy as np
import ollama, pyttsx3, queue, requests, re
from pathlib import Path
from typing import Optional
from collections import deque
from datetime import datetime, timedelta

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIG
# ════════════════════════════════════════════════════════════════════════════════════════════════════
VOICE_MODE = False  # Set to False for text-only mode

ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
WAKE_WORD_PATH = "Hey-Arthur_en_windows_v3_0_0.ppn"
MIC_INDEX, SAMPLE_RATE, CHUNK = 1, 16000, 512
SILENCE_THRESH, SILENCE_TIME, MIN_SPEECH = 500, 1.5, 0.3
OLLAMA_MODEL, MAX_TOKENS, TEMP = "llama3.2:3b", 150, 0.7
VOICE_RATE = 180

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as environment variable
OPENAI_MODEL = "gpt-4o-mini"

# Weather API (OpenWeatherMap - free tier)
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Get free key at openweathermap.org
WEATHER_LOCATION = "Noels Pond,CA"  # Ronan's location

USER_INFO = """Name: Ronan
Age: 13
Location: Noels Pond, Newfoundland, Canada

Current Date/Time: {current_time}

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
        self.history = deque(maxlen=100)
        self.wake_queue = queue.Queue()
        self.interrupt = threading.Event()
        self.speaking = False
        self.audio_lock = threading.Lock()
        self.voice_mode = VOICE_MODE
        
        # AI Mode Selection
        self.use_openai = False
        self.ai_mode = "Checking..."
        
        # Alarms & Timers
        self.alarms = []  # List of (datetime, label) tuples
        self.timers = []  # List of (end_time, label) tuples
        self.alarm_lock = threading.Lock()
        
        # Check AI availability
        self._setup_ai()
        
        if self.voice_mode:
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
            
            print(f"✅ Arthur ready (Voice Mode) - Using {self.ai_mode}!")
        else:
            self.porcupine = None
            self.whisper = None
            self.pa = None
            self.stream = None
            print(f"✅ Arthur ready (Text Mode) - Using {self.ai_mode}!")
        
        # Start alarm/timer monitor
        threading.Thread(target=self.monitor_alarms_timers, daemon=True).start()
    
    def _setup_ai(self):
        """Check internet and AI availability, prioritize OpenAI if available"""
        # First check if OpenAI is available (requires API key and internet)
        if OPENAI_API_KEY:
            if self._check_openai():
                self.use_openai = True
                self.ai_mode = "OpenAI GPT-4o-mini (Online)"
                print("✅ OpenAI: Connected")
                return
            else:
                print("⚠️ OpenAI: Not available (no internet or invalid API key)")
        else:
            print("⚠️ OpenAI: API key not set")
        
        # Fall back to Ollama
        if self._check_ollama():
            self.use_openai = False
            self.ai_mode = "Ollama (Offline)"
            print(f"✅ Ollama: {OLLAMA_MODEL}")
        else:
            print("❌ Error: Neither OpenAI nor Ollama available!")
            print("   - For OpenAI: Set OPENAI_API_KEY environment variable and connect to internet")
            print("   - For Ollama: Start with 'ollama serve'")
            exit(1)
    
    def _check_openai(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            # First do a quick internet check
            test_response = requests.get("https://www.google.com", timeout=3)
            if test_response.status_code != 200:
                return False
            
            # Then check OpenAI API
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Use a simpler endpoint that's faster
            data = {
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            # Accept both 200 (success) and 401 (invalid key but API is reachable)
            return response.status_code in [200, 401]
        except requests.exceptions.Timeout:
            print("   (Timeout checking OpenAI API)")
            return False
        except requests.exceptions.ConnectionError:
            print("   (No internet connection)")
            return False
        except Exception as e:
            print(f"   (Error: {e})")
            return False
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            ollama.list()
            return True
        except:
            return False
    
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
    
    def monitor_alarms_timers(self):
        """Background alarm/timer checker"""
        while True:
            try:
                now = datetime.now()
                
                with self.alarm_lock:
                    # Check alarms
                    triggered_alarms = []
                    for alarm_time, label in self.alarms:
                        if now >= alarm_time:
                            triggered_alarms.append((alarm_time, label))
                    
                    for alarm in triggered_alarms:
                        self.alarms.remove(alarm)
                        msg = f"⏰ ALARM! {alarm[1]}" if alarm[1] else "⏰ ALARM!"
                        print(f"\n{msg}")
                        self.play_alarm_sound()
                        self.speak(msg)
                    
                    # Check timers
                    triggered_timers = []
                    for end_time, label in self.timers:
                        if now >= end_time:
                            triggered_timers.append((end_time, label))
                    
                    for timer in triggered_timers:
                        self.timers.remove(timer)
                        msg = f"⏱️ TIMER DONE! {timer[1]}" if timer[1] else "⏱️ TIMER DONE!"
                        print(f"\n{msg}")
                        self.play_alarm_sound()
                        self.speak(msg)
                
                time.sleep(1)  # Check every second
            except Exception as e:
                print(f"⚠️ Alarm monitor error: {e}")
                time.sleep(1)
    
    def word_to_number(self, text: str) -> str:
        """Convert word numbers to digits"""
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'twenty-one': '21', 'twenty-two': '22', 'twenty-three': '23', 'twenty-four': '24',
            'twenty-five': '25', 'twenty-six': '26', 'twenty-seven': '27', 'twenty-eight': '28',
            'twenty-nine': '29', 'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
            'a': '1', 'an': '1'
        }
        
        # Replace word numbers with digits
        for word, num in word_to_num.items():
            text = re.sub(r'\b' + word + r'\b', num, text)
        
        return text
    
    def parse_time(self, text: str) -> Optional[datetime]:
        """Parse time expressions like '7:30 AM', 'fifteen forty-five', '7 PM'"""
        text = self.word_to_number(text.strip()).upper()
        
        # Try HH:MM AM/PM format
        match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', text)
        if match:
            hour, minute, period = match.groups()
            hour = int(hour)
            minute = int(minute)
            
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            
            target = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= datetime.now():
                target += timedelta(days=1)
            return target
        
        # Try HH AM/PM format
        match = re.search(r'(\d{1,2})\s*(AM|PM)', text)
        if match:
            hour, period = match.groups()
            hour = int(hour)
            
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            
            target = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
            if target <= datetime.now():
                target += timedelta(days=1)
            return target
        
        # Try 24-hour format HH:MM
        match = re.search(r'(\d{1,2}):(\d{2})', text)
        if match:
            hour, minute = match.groups()
            hour = int(hour)
            minute = int(minute)
            
            target = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= datetime.now():
                target += timedelta(days=1)
            return target
        
        return None
    
    def parse_duration(self, text: str) -> Optional[int]:
        """Parse duration in seconds from text like '5 minutes', 'two hours', '30 seconds'"""
        text = self.word_to_number(text.lower())
        total_seconds = 0
        
        # Hours
        match = re.search(r'(\d+)\s*(?:hour|hr)s?', text)
        if match:
            total_seconds += int(match.group(1)) * 3600
        
        # Minutes
        match = re.search(r'(\d+)\s*(?:minute|min)s?', text)
        if match:
            total_seconds += int(match.group(1)) * 60
        
        # Seconds
        match = re.search(r'(\d+)\s*(?:second|sec)s?', text)
        if match:
            total_seconds += int(match.group(1))
        
        return total_seconds if total_seconds > 0 else None
    
    def play_alarm_sound(self):
        """Play alarm sound using system beep"""
        try:
            # Try to use winsound on Windows
            import winsound
            # Play 3 beeps
            for _ in range(3):
                winsound.Beep(1000, 500)  # 1000 Hz for 500ms
                time.sleep(0.2)
        except:
            # Fallback: print bell character (works on most terminals)
            for _ in range(3):
                print('\a', end='', flush=True)
                time.sleep(0.5)
    
    def get_weather(self) -> str:
        """Get weather information"""
        if not WEATHER_API_KEY:
            return "Weather API key not set. Get a free key at openweathermap.org and set OPENWEATHER_API_KEY environment variable."
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={WEATHER_LOCATION}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                feels = data['main']['feels_like']
                desc = data['weather'][0]['description']
                humidity = data['main']['humidity']
                
                return f"It's {temp:.0f}°C in Noels Pond, feels like {feels:.0f}°C. {desc.capitalize()}. Humidity {humidity}%."
            else:
                return "Couldn't fetch weather right now. Check your API key or internet connection."
        except Exception as e:
            return f"Weather check failed: {str(e)}"
    
    def handle_command(self, text: str) -> Optional[str]:
        """Handle special commands (alarms, timers, weather)"""
        lower = text.lower()
        
        # Time query
        if any(phrase in lower for phrase in ['what time', 'current time', 'time is it', 'what\'s the time']):
            current_time = datetime.now().strftime("%I:%M %p")
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return f"It's {current_time} on {current_date}"
        
        # Date query
        if any(phrase in lower for phrase in ['what date', 'what day', 'today\'s date', 'what\'s the date']):
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"
        
        # Reset/Clear history
        if any(phrase in lower for phrase in ['reset history', 'clear history', 'forget everything', 'clear memory', 'reset memory']):
            count = len(self.history)
            self.history.clear()
            return f"Memory cleared! Forgot {count} conversation(s)." if count > 0 else "Memory was already empty."
        
        # Weather
        if any(word in lower for word in ['weather', 'temperature', 'forecast', 'outside']):
            return self.get_weather()
        
        # Set alarm
        if 'alarm' in lower and ('set' in lower or 'create' in lower or 'for' in lower or 'at' in lower):
            alarm_time = self.parse_time(text)
            if alarm_time:
                label = ""
                if 'for' in lower:
                    parts = lower.split('for', 1)
                    if len(parts) > 1:
                        label_part = parts[1].strip().replace('alarm', '').strip()
                        # Remove time-related words and number words
                        time_words = ['am', 'pm', 'o\'clock', 'oclock']
                        number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                                      'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 
                                      'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
                                      'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five',
                                      'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine', 'a', 'an']
                        for word in time_words + number_words:
                            label_part = re.sub(r'\b' + word + r'\b', '', label_part, flags=re.IGNORECASE)
                        # Remove digits and colons
                        label_part = re.sub(r'[\d:]+', '', label_part)
                        label = label_part.strip().strip(',').strip()
                
                with self.alarm_lock:
                    self.alarms.append((alarm_time, label))
                
                time_str = alarm_time.strftime("%I:%M %p")
                if label:
                    return f"Alarm set for {time_str} - {label}"
                return f"Alarm set for {time_str}"
            else:
                return "Couldn't parse that time. Try like '7:30 AM' or '3 PM'"
        
        # List alarms
        if 'alarm' in lower and ('list' in lower or 'show' in lower or 'what' in lower):
            with self.alarm_lock:
                if not self.alarms:
                    return "No alarms set"
                
                msg = f"You have {len(self.alarms)} alarm(s): "
                alarm_list = []
                for alarm_time, label in sorted(self.alarms):
                    time_str = alarm_time.strftime("%I:%M %p")
                    if label:
                        alarm_list.append(f"{time_str} ({label})")
                    else:
                        alarm_list.append(time_str)
                return msg + ", ".join(alarm_list)
        
        # Clear alarms
        if 'alarm' in lower and ('clear' in lower or 'delete' in lower or 'remove' in lower or 'cancel' in lower):
            # Check if user wants to clear a specific alarm
            alarm_time = self.parse_time(text)
            if alarm_time and 'all' not in lower:
                with self.alarm_lock:
                    # Find and remove alarm matching the time
                    for i, (atime, label) in enumerate(self.alarms):
                        if atime.hour == alarm_time.hour and atime.minute == alarm_time.minute:
                            self.alarms.pop(i)
                            time_str = atime.strftime("%I:%M %p")
                            return f"Cancelled alarm at {time_str}"
                    return "No alarm found at that time"
            
            # Clear all alarms
            with self.alarm_lock:
                count = len(self.alarms)
                self.alarms.clear()
                return f"Cleared {count} alarm(s)" if count > 0 else "No alarms to clear"
        
        # Set timer
        if 'timer' in lower and ('set' in lower or 'create' in lower or 'for' in lower or 'start' in lower):
            duration = self.parse_duration(text)
            if duration:
                label = ""
                if 'for' in lower:
                    # Extract label after 'for'
                    parts = text.lower().split('for')
                    if len(parts) > 1:
                        # Remove time-related words to get label
                        label_part = parts[-1]
                        # Remove time words
                        time_words = ['minutes', 'minute', 'min', 'hours', 'hour', 'hr', 'seconds', 'second', 'sec', 'timer']
                        # Remove number words
                        number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                                      'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 
                                      'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
                                      'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five',
                                      'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine', 'a', 'an']
                        # Remove all time and number words
                        for word in time_words + number_words:
                            label_part = re.sub(r'\b' + word + r'\b', '', label_part, flags=re.IGNORECASE)
                        # Also remove digits
                        label_part = re.sub(r'\b\d+\b', '', label_part)
                        label = label_part.strip().strip(',').strip()
                
                end_time = datetime.now() + timedelta(seconds=duration)
                
                with self.alarm_lock:
                    self.timers.append((end_time, label))
                
                # Format duration nicely
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                seconds = duration % 60
                
                time_parts = []
                if hours > 0:
                    time_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes > 0:
                    time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                if seconds > 0:
                    time_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
                
                duration_str = " ".join(time_parts)
                
                if label:
                    return f"Timer set for {duration_str} - {label}"
                return f"Timer set for {duration_str}"
            else:
                return "Couldn't parse that duration. Try like '5 minutes' or '2 hours 30 minutes'"
        
        # List timers
        if 'timer' in lower and ('list' in lower or 'show' in lower or 'what' in lower or 'check' in lower):
            with self.alarm_lock:
                if not self.timers:
                    return "No timers running"
                
                msg = f"You have {len(self.timers)} timer(s): "
                timer_list = []
                for end_time, label in sorted(self.timers):
                    remaining = (end_time - datetime.now()).total_seconds()
                    if remaining > 0:
                        mins = int(remaining // 60)
                        secs = int(remaining % 60)
                        time_str = f"{mins}m {secs}s"
                        if label:
                            timer_list.append(f"{time_str} ({label})")
                        else:
                            timer_list.append(time_str)
                return msg + ", ".join(timer_list)
        
        # Clear timers
        if 'timer' in lower and ('clear' in lower or 'delete' in lower or 'remove' in lower or 'cancel' in lower or 'stop' in lower):
            # Check if user specified a number (e.g., "cancel timer 1" or "stop the first timer")
            match = re.search(r'(?:timer\s*)?(?:number\s*)?(\d+|first|second|third|last)', lower)
            if match and 'all' not in lower:
                position_str = match.group(1)
                
                # Convert position to index
                position_map = {'first': 0, 'second': 1, 'third': 2, 'last': -1}
                if position_str in position_map:
                    index = position_map[position_str]
                else:
                    index = int(position_str) - 1  # Convert to 0-based index
                
                with self.alarm_lock:
                    if not self.timers:
                        return "No timers to cancel"
                    
                    # Sort timers by end time for consistent ordering
                    sorted_timers = sorted(self.timers)
                    
                    if index < 0:
                        index = len(sorted_timers) + index
                    
                    if 0 <= index < len(sorted_timers):
                        timer_to_remove = sorted_timers[index]
                        self.timers.remove(timer_to_remove)
                        
                        remaining = (timer_to_remove[0] - datetime.now()).total_seconds()
                        mins = int(remaining // 60)
                        secs = int(remaining % 60)
                        
                        if timer_to_remove[1]:
                            return f"Cancelled timer: {timer_to_remove[1]} ({mins}m {secs}s remaining)"
                        return f"Cancelled timer with {mins}m {secs}s remaining"
                    else:
                        return f"Timer number {index + 1} doesn't exist"
            
            # Clear all timers
            with self.alarm_lock:
                count = len(self.timers)
                self.timers.clear()
                return f"Cleared {count} timer(s)" if count > 0 else "No timers to clear"
        
        return None
    
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
    
    def ask_openai(self, prompt: str) -> str:
        """Get response from OpenAI"""
        try:
            current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            
            msgs = [{
                "role": "system",
                "content": f"You are Arthur, Ronan's AI. Be casual, witty, concise (1-2 sentences).\n\n{USER_INFO.format(current_time=current_time)}"
            }]
            
            for entry in list(self.history)[-3:]:
                msgs.append({"role": "user", "content": entry["user"]})
                msgs.append({"role": "assistant", "content": entry["assistant"]})
            
            msgs.append({"role": "user", "content": prompt})
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": OPENAI_MODEL,
                "messages": msgs,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMP
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content'].strip()
                self.history.append({"user": prompt, "assistant": reply})
                return reply
            else:
                print(f"❌ OpenAI error: {response.status_code}")
                # Fall back to Ollama if OpenAI fails
                if self._check_ollama():
                    self.speak("Falling back to Ollama")
                    print("   ⚠️ Falling back to Ollama...")
                    self.use_openai = False
                    self.ai_mode = "Ollama (Fallback)"
                    return self.ask_ollama(prompt)
                return "OpenAI request failed. Check your connection or API key."
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
            # Fall back to Ollama if OpenAI fails
            if self._check_ollama():
                self.speak("Falling back to Ollama")
                print("   ⚠️ Falling back to Ollama...")
                self.use_openai = False
                self.ai_mode = "Ollama (Fallback)"
                return self.ask_ollama(prompt)
            return "Something went wrong with OpenAI."
    
    def ask_ollama(self, prompt: str) -> str:
        """Get response from Ollama"""
        try:
            current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            
            msgs = [{
                "role": "system",
                "content": f"You are Arthur, Ronan's AI. Be casual, witty, concise (1-2 sentences).\n\n{USER_INFO.format(current_time=current_time)}"
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
            print(f"❌ Ollama error: {e}")
            return "Something went wrong with Ollama. Is it running?"
    
    def ask(self, prompt: str) -> str:
        """Get AI response"""
        # Check for special commands first
        command_response = self.handle_command(prompt)
        if command_response:
            return command_response
        
        p = prompt.lower().replace('.', '').replace(',', '')
        
        # Custom responses
        for trigger, resp in CUSTOM_RESPONSES.items():
            if trigger in p:
                return resp
        
        print(f"🤔 Thinking... ({self.ai_mode})")
        
        # Route to appropriate AI
        if self.use_openai:
            return self.ask_openai(prompt)
        else:
            return self.ask_ollama(prompt)
    
    def speak(self, text: str):
        """Text to speech - Fixed for Windows"""
        print(f"💬 Arthur: {text}")
        
        if not self.voice_mode:
            return
        
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
    
    def run_text_mode(self):
        """Text-only interaction loop"""
        print("💬 Type your messages below (Ctrl+C to exit)\n")
        print("📝 Commands: 'set alarm for 7:30 AM', 'set timer for 5 minutes', 'what's the weather', 'clear history'\n")
        
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("👋 Goodbye!")
                        break
                    
                    response = self.ask(user_input)
                    self.speak(response)
                    print(f"({len(self.history)} in memory)\n")
                    
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    def run_voice_mode(self):
        """Voice interaction loop"""
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
    
    def run(self):
        """Main loop - chooses mode based on VOICE_MODE setting"""
        try:
            if self.voice_mode:
                self.run_voice_mode()
            else:
                self.run_text_mode()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        if self.voice_mode and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
        print("👋 Goodbye!")

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# 🚀 START
# ════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    VoiceAssistant().run()