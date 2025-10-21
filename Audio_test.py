import pyaudio
import struct
import math

print("🎤 Audio Device Diagnostic Tool\n")
print("=" * 50)

pa = pyaudio.PyAudio()

# List all audio devices
print("\n📋 Available Audio Devices:\n")
info = pa.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

input_devices = []

for i in range(0, numdevices):
    try:
        device_info = pa.get_device_info_by_host_api_device_index(0, i)
        device_name = device_info.get('name')
        max_input_channels = device_info.get('maxInputChannels')
        
        if max_input_channels > 0:
            input_devices.append(i)
            print(f"   [{i}] {device_name}")
            print(f"       Channels: {max_input_channels}")
            print(f"       Sample Rate: {int(device_info.get('defaultSampleRate'))} Hz")
            print()
    except Exception as e:
        continue

if not input_devices:
    print("❌ No input devices found!")
    pa.terminate()
    exit(1)

print("=" * 50)
print(f"\n✅ Found {len(input_devices)} input device(s)")

# Show default device
try:
    default_device = pa.get_default_input_device_info()
    default_index = default_device['index']
    print(f"\n⭐ Default microphone: [{default_index}] {default_device['name']}")
except:
    print("\n⚠️ Could not determine default microphone")

# Test default device
print("\n🔍 Testing Default Microphone...\n")

try:
    # Try to open default device
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=512,
        input_device_index=None  # Use default
    )
    
    print("✅ Default microphone opened successfully!")
    print("🎙️ Speak into your microphone for 3 seconds...")
    print("   (Testing audio level detection)\n")
    
    # Read audio for 3 seconds and check levels
    max_amplitude = 0
    total_samples = 0
    
    for i in range(0, int(16000 / 512 * 3)):  # 3 seconds
        try:
            data = stream.read(512, exception_on_overflow=False)
            
            # Calculate amplitude
            audio_data = struct.unpack(str(512) + 'h', data)
            amplitude = sum(abs(x) for x in audio_data) / len(audio_data)
            max_amplitude = max(max_amplitude, amplitude)
            total_samples += 1
            
            # Visual feedback
            if i % 10 == 0:  # Update every ~0.3 seconds
                bars = int(amplitude / 100)
                print(f"   Volume: {'█' * min(bars, 50)} {int(amplitude)}", end='\r')
                
        except Exception as e:
            print(f"\n⚠️ Read error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    
    print("\n")
    print("=" * 50)
    print("\n📊 Results:")
    print(f"   Max amplitude detected: {int(max_amplitude)}")
    
    if max_amplitude < 10:
        print("\n❌ PROBLEM: Very low audio levels detected!")
        print("   Possible issues:")
        print("   • Microphone is muted")
        print("   • Wrong microphone selected")
        print("   • Microphone not plugged in")
        print("   • Microphone permissions not granted")
        print("\n   💡 Try:")
        print("   • Check Windows Sound Settings")
        print("   • Ensure microphone is selected as default recording device")
        print("   • Test microphone in Windows Settings > System > Sound")
    elif max_amplitude < 100:
        print("\n⚠️ WARNING: Low audio levels")
        print("   • Microphone might be too far away")
        print("   • Try speaking louder")
        print("   • Check microphone volume in Windows settings")
    else:
        print("\n✅ Good audio levels detected!")
        print("   Your microphone is working properly!")
    
except Exception as e:
    print(f"\n❌ Failed to open default microphone: {e}")
    print("\n🔧 Troubleshooting:")
    print("   • Check if microphone is connected")
    print("   • Grant microphone permissions to Python")
    print("   • Try running as Administrator")
    print("   • Check Windows Privacy Settings > Microphone")

pa.terminate()

print("\n" + "=" * 50)
print("\n💡 To use a specific device in Arthur:")
print("   In your Arthur code, change:")
print("   MICROPHONE_INDEX = None")
print("   To:")
print("   MICROPHONE_INDEX = X  # Where X is your device number from above")
print("\n" + "=" * 50)