import config

class FatigueMetrics:
    """
    F = 1/2 * (1/2 * F_blink + 3/10 * F_yawn + 1/5 * F_nod + PERCLOS)
    """
    def __init__(self, window_size=config.FATIGUE_WINDOW_SIZE, fps=config.FPS_DEFAULT):
        self.window_size = window_size
        self.fps = fps
        self.history = []

        self.T_EAR = config.T_EAR
        self.T_MAR = config.T_MAR
        self.T_PITCH = config.T_PITCH
        
        self.is_blinking = False
        self.is_yawning = False
        self.is_nodding = False

        self.blink_count = 0
        self.yawn_count = 0
        self.nod_count = 0

    def update(self, ear, mar, pitch):
        state = {
            "ear": ear,
            "mar": mar,
            "pitch": pitch,
            "blink_event": 0,
            "yawn_event": 0,
            "nod_event": 0
        }

        # Blink
        if ear < self.T_EAR:
            self.is_blinking = True
        elif ear >= self.T_EAR and self.is_blinking:
            self.is_blinking = False
            state["blink_event"] = 1
            self.blink_count += 1

        # Yawn
        if mar > self.T_MAR:
            self.is_yawning = True
        elif mar <= self.T_MAR and self.is_yawning:
            self.is_yawning = False
            state["yawn_event"] = 1
            self.yawn_count += 1

        # Nod
        if pitch < self.T_PITCH:
            self.is_nodding = True
        elif pitch >= self.T_PITCH and self.is_nodding:
            self.is_nodding = False
            state["nod_event"] = 1
            self.nod_count += 1

        self.history.append(state)

        # Sliding window
        if len(self.history) > self.window_size:
            removed = self.history.pop(0)
            self.blink_count -= removed["blink_event"]
            self.yawn_count -= removed["yawn_event"]
            self.nod_count -= removed["nod_event"]

    def get_metrics(self):
        if not self.history:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # PERCLOS
        closed_frames = sum(1 for s in self.history if s["ear"] < self.T_EAR)
        perclos = closed_frames / len(self.history)

        # Fast blink count indicates alertness/stress
        f_blink = self.blink_count / 4.0
        
        # Yawn and Nod (requires 2 events in the window to reach max 1.0, 1 event equals 0.5)
        f_yawn = min(1.0, self.yawn_count / 2.0)
        f_nod = min(1.0, self.nod_count / 2.0)

        # F_score (Fatigue) = Combined weighted score
        # 70% weight on eye closure (PERCLOS)
        # 1 yawn -> +0.075 points
        # 1 nod -> +0.075 points
        # Closing eyes 30% of the time (PERCLOS 0.3) -> 0.7 * 0.3 = 0.21 -> Alarm (Threshold 0.20)
        f_combined = (0.7 * perclos) + (0.15 * f_yawn) + (0.15 * f_nod)

        return f_combined, perclos, f_blink, f_yawn, f_nod
