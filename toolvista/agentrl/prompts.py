"""Identical screenshot/action instructions for SFT, gate, and RL."""
SYSTEM_PROMPT = '''You control a browser using screenshots. Follow the task shown below.
Return exactly one JSON object per turn, without markdown or explanation:
{"action":"click","x":0.5,"y":0.5} (coordinates normalized to the full screenshot, 0..1)
{"action":"type","text":"hello"} (type into the currently focused field)
{"action":"scroll","dx":0,"dy":300} (pixel deltas, each -1000..1000)
{"action":"press","key":"Enter"}
{"action":"finish","status":"success"} or {"action":"finish","status":"failure"}
Allowed keys: Enter, Tab, Escape, Backspace, Delete, Space, ArrowUp, ArrowDown,
ArrowLeft, ArrowRight, Home, End, Control+A.
Use the next screenshot to check the effect of your action. Only the environment
decides whether the task succeeded. Do not assume that saying success completes it.'''


def observation_text(observation: dict) -> str:
    return f"Task: {observation['goal']}\nStep: {observation['step']}\nFeedback: {observation.get('feedback', 'ready')}"


def observation_message(observation: dict) -> dict:
    return {"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": observation_text(observation)}]}
