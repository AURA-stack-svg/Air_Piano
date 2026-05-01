import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pygame
import time
from collections import deque

class ImprovedAirPiano:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)
        
        pygame.init()
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        self.setup_piano_layout()
        
        self.white_key_color = (255, 255, 255)
        self.black_key_color = (40, 40, 40)
        self.pressed_white_color = (100, 200, 255)
        self.pressed_black_color = (150, 100, 255)
        self.finger_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                             (255, 255, 100), (255, 100, 255)]
        
        self.finger_tracking = {}
        self.tracking_history_length = 6
        self.velocity_threshold = 15
        self.press_cooldown = 0.15
        self.depth_threshold = 30
        
        self.calibration_state = "NOT_STARTED"  # States: NOT_STARTED, CALIBRATING, DONE
        self.calibration_start_time = None
        self.calibration_duration = 3  # seconds to hold hand in position
        self.initial_hand_position = None
        self.hand_position_tolerance = 100  # pixels
        
        self.generate_piano_sounds()
        self.setup_songs()
        
        self.current_song = 0
        self.current_note = 0
        self.song_playing = False
        self.score = 0
        self.last_played_time = {}
        
        self.frame_count = 0
        self.fps_counter = time.time()
        self.current_fps = 0
        
    def setup_piano_layout(self):
        screen_width = 1280
        screen_height = 720

        # Make keys bigger
        self.white_key_width = 128  # was 90
        self.white_key_height = 400  # was 300
        self.black_key_width = 80    # was 55
        self.black_key_height = 270  # was 200

        total_white_keys = 10  # was 14
        total_piano_width = total_white_keys * self.white_key_width
        self.piano_start_x = (screen_width - total_piano_width) // 2
        self.piano_start_y = screen_height - self.white_key_height - 50

        self.white_keys = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4', 'D4', 'E4']  # 10 keys
        self.black_keys = ['C#3', 'D#3', 'F#3', 'G#3', 'A#3', 'C#4', 'D#4']  # adjust to fit

        self.setup_key_positions()
    
    def setup_key_positions(self):
        self.white_key_rects = []
        for i, key in enumerate(self.white_keys):
            rect = {
                'key': key,
                'x': self.piano_start_x + i * self.white_key_width,
                'y': self.piano_start_y,
                'width': self.white_key_width,
                'height': self.white_key_height,
                'pressed': False
            }
            self.white_key_rects.append(rect)

        self.black_key_rects = []
        # Adjust black key positions for 10 white keys
        black_key_positions = [0, 1, 3, 4, 5, 7, 8]  # fits 7 black keys for 10 whites

        for i, key in enumerate(self.black_keys):
            if i < len(black_key_positions):
                white_key_index = black_key_positions[i]
                rect = {
                    'key': key,
                    'x': self.piano_start_x + white_key_index * self.white_key_width +
                         self.white_key_width - self.black_key_width // 2,
                    'y': self.piano_start_y,
                    'width': self.black_key_width,
                    'height': self.black_key_height,
                    'pressed': False
                }
                self.black_key_rects.append(rect)
    
    def generate_piano_sounds(self):
        self.sounds = {}
        
        note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        sample_rate = 44100
        duration = 1.8
        
        for octave in [3, 4]:
            for note, base_freq in note_frequencies.items():
                freq = base_freq * (2 ** (octave - 4))
                key_name = f"{note}{octave}"
                
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                
                wave = np.zeros_like(t)
                harmonics = [1.0, 0.5, 0.25, 0.125, 0.0625]
                
                for i, amplitude in enumerate(harmonics):
                    harmonic_freq = freq * (i + 1)
                    wave += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
                
                attack_time = 0.03
                decay_time = 0.2
                sustain_level = 0.3
                release_time = 1.4
                
                attack_samples = int(attack_time * sample_rate)
                decay_samples = int(decay_time * sample_rate)
                release_samples = int(release_time * sample_rate)
                sustain_samples = len(t) - attack_samples - decay_samples - release_samples
                
                envelope = np.ones_like(t)
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
                envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level
                envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
                
                wave = wave * envelope * 0.25
                wave = np.clip(wave, -1, 1)
                wave_int = (wave * 32767).astype(np.int16)
                stereo_wave = np.column_stack((wave_int, wave_int))
                
                try:
                    self.sounds[key_name] = pygame.mixer.Sound(stereo_wave)
                except:
                    pass
    
    def setup_songs(self):
        self.songs = [
            {
                'name': 'Twinkle Twinkle Little Star',
                'notes': ['C4', 'C4', 'G4', 'G4', 'A4', 'A4', 'G4',
                         'F4', 'F4', 'E4', 'E4', 'D4', 'D4', 'C4']
            },
            {
                'name': 'Mary Had a Little Lamb',
                'notes': ['E4', 'D4', 'C4', 'D4', 'E4', 'E4', 'E4',
                         'D4', 'D4', 'D4', 'E4', 'G4', 'G4']
            },
            {
                'name': 'Happy Birthday',
                'notes': ['C4', 'C4', 'D4', 'C4', 'F4', 'E4',
                         'C4', 'C4', 'D4', 'C4', 'G4', 'F4']
            }
        ]
    
    def track_finger(self, finger_id, position):
        if finger_id not in self.finger_tracking:
            self.finger_tracking[finger_id] = {
                'positions': deque(maxlen=self.tracking_history_length),
                'is_pressing': False,
                'last_press_time': 0,
                'press_y': 0
            }
        
        tracking = self.finger_tracking[finger_id]
        tracking['positions'].append(position)
        
        if len(tracking['positions']) >= 3:
            recent_positions = list(tracking['positions'])[-3:]
            velocity_y = recent_positions[-1][1] - recent_positions[-3][1]
            return velocity_y
        
        return 0
    
    def detect_press(self, finger_id, position, velocity):
        current_time = time.time()
        tracking = self.finger_tracking[finger_id]
        
        key_under_finger = self.get_key_at_position(position[0], position[1])
        
        if not key_under_finger:
            tracking['is_pressing'] = False
            return None
        
        if current_time - tracking['last_press_time'] < self.press_cooldown:
            return None
        
        press_depth = position[1] - self.piano_start_y
        
        if (not tracking['is_pressing'] and 
            velocity > self.velocity_threshold and
            press_depth > self.depth_threshold):
            
            tracking['is_pressing'] = True
            tracking['last_press_time'] = current_time
            tracking['press_y'] = position[1]
            return key_under_finger
        
        elif (tracking['is_pressing'] and 
              (velocity < -self.velocity_threshold or 
               position[1] < tracking['press_y'] - 20)):
            tracking['is_pressing'] = False
        
        return None
    
    def get_key_at_position(self, x, y):
        for key_rect in self.black_key_rects:
            if (key_rect['x'] <= x <= key_rect['x'] + key_rect['width'] and
                key_rect['y'] <= y <= key_rect['y'] + key_rect['height']):
                return key_rect['key']
        
        for key_rect in self.white_key_rects:
            if (key_rect['x'] <= x <= key_rect['x'] + key_rect['width'] and
                key_rect['y'] <= y <= key_rect['y'] + key_rect['height']):
                return key_rect['key']
        
        return None
    
    def play_note(self, note):
        current_time = time.time()
        
        if note in self.last_played_time:
            if current_time - self.last_played_time[note] < 0.1:
                return
        
        self.last_played_time[note] = current_time
        
        if note in self.sounds:
            try:
                self.sounds[note].play()
                
                if (self.song_playing and 
                    self.current_note < len(self.songs[self.current_song]['notes'])):
                    expected_note = self.songs[self.current_song]['notes'][self.current_note]
                    
                    if note == expected_note:
                        self.score += 10
                        self.current_note += 1
                        
                        if self.current_note >= len(self.songs[self.current_song]['notes']):
                            self.song_playing = False
                            self.score += 100
                            
            except:
                pass
    
    def draw_piano(self, img):
        bg_padding = 15
        bg_rect = (
            self.piano_start_x - bg_padding,
            self.piano_start_y - bg_padding,
            len(self.white_keys) * self.white_key_width + 2 * bg_padding,
            self.white_key_height + 2 * bg_padding
        )
        
        cv2.rectangle(img, 
                     (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
                     (25, 25, 25), cv2.FILLED)
        
        for key_rect in self.white_key_rects:
            color = self.pressed_white_color if key_rect['pressed'] else self.white_key_color
            
            cv2.rectangle(img,
                         (key_rect['x'], key_rect['y']),
                         (key_rect['x'] + key_rect['width'], key_rect['y'] + key_rect['height']),
                         color, cv2.FILLED)
            
            cv2.rectangle(img,
                         (key_rect['x'], key_rect['y']),
                         (key_rect['x'] + key_rect['width'], key_rect['y'] + key_rect['height']),
                         (0, 0, 0), 2)
            
            label = key_rect['key']
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = key_rect['x'] + (key_rect['width'] - text_size[0]) // 2
            text_y = key_rect['y'] + key_rect['height'] - 20
            
            cv2.putText(img, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        for key_rect in self.black_key_rects:
            color = self.pressed_black_color if key_rect['pressed'] else self.black_key_color
            
            cv2.rectangle(img,
                         (key_rect['x'], key_rect['y']),
                         (key_rect['x'] + key_rect['width'], key_rect['y'] + key_rect['height']),
                         color, cv2.FILLED)
            
            cv2.rectangle(img,
                         (key_rect['x'], key_rect['y']),
                         (key_rect['x'] + key_rect['width'], key_rect['y'] + key_rect['height']),
                         (120, 120, 120), 2)
            
            label = key_rect['key']
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = key_rect['x'] + (key_rect['width'] - text_size[0]) // 2
            text_y = key_rect['y'] + key_rect['height'] - 15
            
            cv2.putText(img, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_hands(self, img, hands):
        if not hands:
            return
        
        for hand_idx, hand in enumerate(hands):
            lmList = hand['lmList']
            
            fingertip_indices = [4, 8, 12, 16, 20]
            
            for finger_idx, tip_idx in enumerate(fingertip_indices):
                x, y = lmList[tip_idx][:2]
                finger_id = f"h{hand_idx}_f{finger_idx}"
                
                velocity = self.track_finger(finger_id, (x, y))
                pressed_key = self.detect_press(finger_id, (x, y), velocity)
                
                if pressed_key:
                    self.play_note(pressed_key)
                    self.set_key_state(pressed_key, True)
                
                is_pressing = self.finger_tracking.get(finger_id, {}).get('is_pressing', False)
                
                if is_pressing:
                    cv2.circle(img, (x, y), 14, self.finger_colors[finger_idx], cv2.FILLED)
                    cv2.circle(img, (x, y), 17, (255, 255, 255), 3)
                else:
                    cv2.circle(img, (x, y), 10, self.finger_colors[finger_idx], cv2.FILLED)
                    cv2.circle(img, (x, y), 13, (255, 255, 255), 2)
                
                cv2.putText(img, f"{velocity:.0f}", (x - 15, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def set_key_state(self, key, pressed):
        for key_rect in self.white_key_rects + self.black_key_rects:
            if key_rect['key'] == key:
                key_rect['pressed'] = pressed
                break
    
    def reset_keys(self):
        for key_rect in self.white_key_rects + self.black_key_rects:
            key_rect['pressed'] = False
    
    def draw_ui(self, img):
        cv2.putText(img, "Enhanced Air Piano", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(img, f"FPS: {self.current_fps}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        controls = ["1-3: Songs", "SPACE: Tutorial", "Q: Quit"]
        for i, control in enumerate(controls):
            cv2.putText(img, control, (20, 100 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.song_playing:
            song = self.songs[self.current_song]
            
            cv2.rectangle(img, (400, 20), (1260, 100), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(img, (400, 20), (1260, 100), (255, 255, 255), 2)
            
            cv2.putText(img, f"Playing: {song['name']}", (420, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            progress = f"Note {self.current_note + 1}/{len(song['notes'])}"
            cv2.putText(img, progress, (420, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if self.current_note < len(song['notes']):
                next_note = song['notes'][self.current_note]
                cv2.putText(img, f"Next: {next_note}", (420, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                self.highlight_key(img, next_note)
            
            cv2.putText(img, f"Score: {self.score}", (1100, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        current_song_name = self.songs[self.current_song]['name']
        cv2.putText(img, f"Selected: {current_song_name}", (400, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def highlight_key(self, img, note):
        for key_rect in self.white_key_rects + self.black_key_rects:
            if key_rect['key'] == note:
                pulse = int(100 * (1 + np.sin(time.time() * 6)) / 2) + 100
                highlight_color = (255, pulse, 0)
                
                cv2.rectangle(img,
                             (key_rect['x'] - 3, key_rect['y'] - 3),
                             (key_rect['x'] + key_rect['width'] + 3, 
                              key_rect['y'] + key_rect['height'] + 3),
                             highlight_color, 4)
                break
    
    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_counter >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.fps_counter = current_time
    
    def handle_calibration(self, img, hands):
        """Handle the hand calibration process"""
        if not hands:
            self.calibration_state = "NOT_STARTED"
            self.calibration_start_time = None
            cv2.putText(img, "Place your hand above the keyboard to begin",
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return False
            
        if self.calibration_state == "NOT_STARTED":
            # Start calibration when hand is detected in a good position
            hand = hands[0]
            palm_center = hand['center']
            if self.piano_start_y - 200 <= palm_center[1] <= self.piano_start_y - 100:
                self.calibration_state = "CALIBRATING"
                self.calibration_start_time = time.time()
                self.initial_hand_position = palm_center
        
        elif self.calibration_state == "CALIBRATING":
            hand = hands[0]
            palm_center = hand['center']
            
            # Check if hand is still in position
            if self.initial_hand_position:
                distance = np.sqrt((palm_center[0] - self.initial_hand_position[0])**2 +
                                 (palm_center[1] - self.initial_hand_position[1])**2)
                
                if distance > self.hand_position_tolerance:
                    self.calibration_state = "NOT_STARTED"
                    cv2.putText(img, "Keep your hand steady!", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    time_remaining = self.calibration_duration - (time.time() - self.calibration_start_time)
                    if time_remaining > 0:
                        cv2.putText(img, f"Hold position: {time_remaining:.1f}s", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.calibration_state = "DONE"
        
        return self.calibration_state == "DONE"
    
    def run(self):
        print("Air Piano Started! Press DOWN over keys to play.")
        
        while True:
            success, img = self.cap.read()
            if not success:
                continue
            
            img = cv2.flip(img, 1)
            self.update_fps()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_song = 0
                self.current_note = 0
                self.score = 0
            elif key == ord('2'):
                self.current_song = 1
                self.current_note = 0
                self.score = 0
            elif key == ord('3'):
                self.current_song = 2
                self.current_note = 0
                self.score = 0
            elif key == ord(' '):
                self.song_playing = not self.song_playing
                if self.song_playing:
                    self.current_note = 0
                    self.score = 0
            
            self.reset_keys()
            
            hands, img = self.detector.findHands(img, draw=True, flipType=False)
            
            if self.handle_calibration(img, hands):
                pass
            else:
                self.draw_piano(img)
                self.draw_hands(img, hands)
                self.draw_ui(img)
            
            cv2.imshow("Air Piano", img)
        
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        pygame.quit()

if __name__ == "__main__":
    try:
        piano = ImprovedAirPiano()
        piano.run()
    except Exception as e:
        print(f"Error: {e}")