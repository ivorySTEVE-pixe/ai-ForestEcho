function useTweaks(defaults) {
  const [tweaks, setTweaks] = React.useState(defaults || {});
  const setTweak = (key, value) => {
    setTweaks((prev) => ({ ...prev, [key]: value }));
  };
  return [tweaks, setTweak];
}

function TweaksPanel({ children }) {
  return null;
}

function TweakSection({ label }) {
  return null;
}

function TweakColor({ label, value, onChange }) {
  return null;
}

function TweakToggle({ label, value, onChange }) {
  return null;
}

function TweakSlider({ label, value, min, max, step, onChange }) {
  return null;
}

function TweakText({ label, value, onChange }) {
  return null;
}
