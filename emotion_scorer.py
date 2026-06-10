class MultiEmotionScorer:
    def __init__(self, target_classes: list, window_size: int = 90):
        self.window_size = window_size
        self.target_classes = target_classes
        self.history = {cls: [] for cls in target_classes}

    def update(self, probs: list):
        for cls in self.target_classes:
            score = float(probs[cls]) if len(probs) > cls else 0.0
            self.history[cls].append(score)
            if len(self.history[cls]) > self.window_size:
                self.history[cls].pop(0)

    def get_levels(self) -> dict:
        levels = {}
        for cls, hist in self.history.items():
            if not hist:
                levels[cls] = 0.0
            else:
                levels[cls] = sum(hist) / len(hist)
        return levels

    def reset(self):
        for cls in self.history:
            self.history[cls].clear()


# Backward-compat alias
AngerScorer = MultiEmotionScorer
EmotionScorer = MultiEmotionScorer
