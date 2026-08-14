"""
scent_publisher.py
------------------
A tiny Adafruit IO MQTT publisher for sending detected scent classes to Unity.

Used by inference.py: whenever the classifier confirms a new scent, it calls
publisher.publish(class_name), which sends the class name to an Adafruit IO
feed. The Unity ScentReceiver script is subscribed to that same feed and
reveals the matching object.

    eNose -> inference.py -> [this publisher] -> Adafruit IO -> Unity

Install the dependency once:
    python -m pip install paho-mqtt

The credentials below MUST match the ones in the Unity ScentReceiver Inspector
(same username, same key, same feed name).
"""

import ssl
import paho.mqtt.client as mqtt


class ScentPublisher:
    def __init__(self, username, aio_key, feed_name, use_tls=False, verbose=True):
        self.username = username
        self.feed = f"{username}/feeds/{feed_name}"
        self.verbose = verbose
        self._connected = False

        # paho-mqtt v2 callback API
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(username, aio_key)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        if use_tls:
            self.client.tls_set(cert_reqs=ssl.CERT_NONE)
            port = 8883
        else:
            port = 1883

        try:
            self.client.connect("io.adafruit.com", port, keepalive=60)
            self.client.loop_start()  # background network thread
        except Exception as e:
            print(f"[ScentPublisher] Could not connect to Adafruit IO: {e}")
            print("[ScentPublisher] Detections will still print locally but won't reach Unity.")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._connected = True
            if self.verbose:
                print(f"[ScentPublisher] Connected to Adafruit IO, publishing to '{self.feed}'")
        else:
            print(f"[ScentPublisher] Connect failed (code {reason_code}). Check username/key.")

    def _on_disconnect(self, client, userdata, *args):
        self._connected = False
        if self.verbose:
            print("[ScentPublisher] Disconnected from Adafruit IO.")

    def publish(self, class_name):
        """Send a detected scent class name to Unity. Safe to call even if not connected."""
        try:
            result = self.client.publish(self.feed, class_name, qos=1)
            if self.verbose:
                status = "sent" if result.rc == mqtt.MQTT_ERR_SUCCESS else f"queued (rc={result.rc})"
                print(f"[ScentPublisher] -> Unity: '{class_name}' ({status})")
        except Exception as e:
            print(f"[ScentPublisher] publish error: {e}")

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
