"""Audio device discovery for Windows.

Why this file exists: to hear the *other person* in a Meet/Zoom/Teams call we
must capture what Windows is *playing*, not what the microphone hears. On
Windows that is WASAPI loopback - you open the speaker/output endpoint as if it
were an input.

Two backends are supported because loopback is the single most fragile part of
this app:

  * sounddevice >= 0.5 -- `WasapiSettings(loopback=True)`. Preferred: one
    library covers both microphone and loopback.
  * soundcard         -- pure-ctypes WASAPI wrapper, used as a fallback when
    the installed PortAudio build has no loopback support.

Device enumeration always goes through sounddevice (its list is the one users
recognise from the Windows sound panel); the soundcard backend matches by name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

LOOPBACK = "loopback"
MICROPHONE = "microphone"


@dataclass
class AudioDevice:
    index: int           # sounddevice device index
    name: str
    kind: str            # LOOPBACK | MICROPHONE
    channels: int
    samplerate: int
    hostapi: str = ""
    is_default: bool = False

    @property
    def label(self) -> str:
        tag = "System Audio" if self.kind == LOOPBACK else "Microphone"
        star = " *" if self.is_default else ""
        return "%s - %s%s" % (tag, self.name, star)

    def __str__(self) -> str:  # pragma: no cover - display only
        return "[%d] %s (%d ch @ %d Hz)" % (
            self.index, self.label, self.channels, self.samplerate
        )


class AudioDeviceError(RuntimeError):
    pass


def _sounddevice():
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - import guard
        raise AudioDeviceError(
            "The 'sounddevice' package is not available (%s). "
            "Install it with: pip install sounddevice" % exc
        ) from exc
    return sd


def loopback_supported() -> bool:
    """True when the installed PortAudio exposes WASAPI loopback."""
    try:
        sd = _sounddevice()
    except Exception:
        return False
    if not hasattr(sd, "WasapiSettings"):
        return False
    return _wasapi_settings_accepts_loopback(sd)


def _wasapi_settings_accepts_loopback(sd) -> bool:
    try:
        sd.WasapiSettings(loopback=True)
        return True
    except Exception:
        return False


def soundcard_available() -> bool:
    try:
        import soundcard  # noqa: F401
        return True
    except Exception:
        return False


def _same_endpoint(name: str, other: str) -> bool:
    """Compare device names across host APIs.

    MME truncates names to 31 characters, so "Speakers (Realtek(R) Audio)" can
    appear as "Speakers (Realtek(R) Aud" elsewhere. Comparing prefixes is more
    reliable than comparing indexes.
    """
    if not name or not other:
        return False
    a, b = name.strip().lower(), other.strip().lower()
    if a == b:
        return True
    shortest = min(len(a), len(b), 24)
    return shortest >= 6 and a[:shortest] == b[:shortest]


def list_devices(include_microphones: bool = True) -> list:
    """Every capturable endpoint, loopback first.

    Loopback entries are WASAPI *output* devices; microphone entries are any
    device with input channels. Devices that raise while being queried are
    skipped rather than aborting the whole listing.
    """
    sd = _sounddevice()
    devices = []

    try:
        raw = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        raise AudioDeviceError("Could not query audio devices: %s" % exc) from exc

    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in = default_out = -1

    # The default output index reported by PortAudio usually belongs to the MME
    # host API, not WASAPI, so it will never equal a loopback candidate's index.
    # Match on the device *name* instead - that is stable across host APIs.
    default_output_name = ""
    try:
        if default_out is not None and default_out >= 0:
            default_output_name = raw[default_out]["name"].strip().lower()
    except Exception:
        default_output_name = ""

    wasapi_indexes = set()
    for api in hostapis:
        if "wasapi" in api["name"].lower():
            wasapi_indexes.add(hostapis.index(api))

    seen_loopback = set()
    for index, dev in enumerate(raw):
        try:
            name = dev["name"].strip()
            api_index = dev["hostapi"]
            api_name = hostapis[api_index]["name"]
            samplerate = int(dev["default_samplerate"] or 48000)

            # --- loopback candidates: WASAPI outputs -----------------------
            if dev["max_output_channels"] > 0 and api_index in wasapi_indexes:
                key = name.lower()
                if key not in seen_loopback:
                    seen_loopback.add(key)
                    devices.append(
                        AudioDevice(
                            index=index,
                            name=name,
                            kind=LOOPBACK,
                            channels=min(2, int(dev["max_output_channels"])),
                            samplerate=samplerate,
                            hostapi=api_name,
                            is_default=_same_endpoint(name, default_output_name),
                        )
                    )

            # --- microphones ----------------------------------------------
            if include_microphones and dev["max_input_channels"] > 0:
                devices.append(
                    AudioDevice(
                        index=index,
                        name=name,
                        kind=MICROPHONE,
                        channels=min(2, int(dev["max_input_channels"])),
                        samplerate=samplerate,
                        hostapi=api_name,
                        is_default=(index == default_in),
                    )
                )
        except Exception:
            log.debug("Skipping unreadable device at index %d", index, exc_info=True)

    # Default loopback device first, then the rest of the loopbacks, then mics.
    devices.sort(key=lambda d: (d.kind != LOOPBACK, not d.is_default, d.name.lower()))
    return devices


def resolve_device(spec: str = "", mode: str = LOOPBACK, devices: list = None) -> AudioDevice:
    """Turn an AUDIO_DEVICE string into a concrete device.

    `spec` may be a device index ("7") or a case-insensitive name substring
    ("Speakers"). Empty means "the default device for `mode`".
    """
    devices = devices if devices is not None else list_devices()
    if not devices:
        raise AudioDeviceError(
            "No audio devices were found. Check that Windows sees a playback "
            "device in Settings > System > Sound."
        )

    spec = (spec or "").strip()

    if spec:
        if spec.isdigit():
            index = int(spec)
            for dev in devices:
                if dev.index == index:
                    return dev
            raise AudioDeviceError(
                "AUDIO_DEVICE=%s does not match any device. Run "
                "`python app.py --list-devices` to see valid indexes." % spec
            )
        lowered = spec.lower()
        matches = [d for d in devices if lowered in d.name.lower()]
        preferred = [d for d in matches if d.kind == mode]
        if preferred:
            return preferred[0]
        if matches:
            return matches[0]
        raise AudioDeviceError(
            "No audio device name contains '%s'. Run "
            "`python app.py --list-devices` to see the available names." % spec
        )

    for dev in devices:
        if dev.kind == mode and dev.is_default:
            return dev
    for dev in devices:
        if dev.kind == mode:
            return dev

    # Requested mode unavailable - fall back rather than refusing to start.
    log.warning("No %s device available; falling back to %s", mode, devices[0].kind)
    return devices[0]


def describe_devices() -> str:  # pragma: no cover - CLI helper
    lines = ["", "Available audio devices (* = Windows default)", "-" * 64]
    try:
        for dev in list_devices():
            lines.append("  %s" % dev)
    except AudioDeviceError as exc:
        lines.append("  ERROR: %s" % exc)
    lines.append("-" * 64)
    lines.append(
        "Pick a 'System Audio' entry to hear the other person in a call; pick a "
        "'Microphone' entry to hear yourself."
    )
    lines.append("Set it in .env, e.g.  AUDIO_DEVICE=7   or   AUDIO_DEVICE=Speakers")
    lines.append("")
    return "\n".join(lines)
