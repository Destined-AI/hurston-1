# Hurston-1 in Dograh

[Dograh](https://github.com/dograh-hq/dograh) is an open-source voice-agent platform. It can use any TTS server
that speaks OpenAI's `/v1/audio/speech` API, and `serve.py` in this repo is one. Run it on your own GPU, point
Dograh at it, and your agents talk in Southern American, Nigerian, West, East or Southern African English.

## 1. Run the Hurston-1 server

On a machine with an NVIDIA GPU (about 8 GB of GPU memory for the model, plus cache):

```bash
git clone https://github.com/Destined-AI/hurston-1 && cd hurston-1
pip install -r requirements.txt -r requirements-serve.txt
hf auth login                 # once: accept the terms on huggingface.co/destined/hurston-1 first
python serve.py               # listens on http://0.0.0.0:8890/v1
```

Check it:

```bash
curl -s localhost:8890/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"hurston-1","voice":"southern-female","input":"Hey there, thanks for calling.","response_format":"wav"}' > hello.wav
```

With vLLM installed (it is in `requirements-serve.txt`) audio starts streaming about 0.2 s after the request and is
generated about twice as fast as real time. Without vLLM the server falls back to plain transformers, which only
returns each sentence once it is finished; fine for trying it out, too slow for live calls.

Options: `--port`, `--model /path/to/weights` (or `HURSTON_MODEL`), `--gpu-memory-utilization` (default 0.3),
`--api-key` (or `HURSTON_API_KEY`) to require `Authorization: Bearer <key>`.

## 2. Point Dograh at it

In Dograh, open the model configuration for your agent and set the voice (TTS) provider:

| Field | Value |
|---|---|
| Provider | **Speaches** (Dograh's option for self-hosted OpenAI-compatible TTS) |
| Base URL | `http://<server>:8890/v1` |
| Model | `hurston-1` |
| Voice | a voice name from the list below, e.g. `southern-female` |
| Speed | 1.0 (see below) |
| API key | blank, unless you started the server with `--api-key` |

What `<server>` is depends on where Dograh runs:

- **Dograh in Docker on the same machine:** `localhost` inside the container is the container itself. Use the
  machine's LAN address (e.g. `http://192.168.1.20:8890/v1`), or add `extra_hosts: ["host.docker.internal:host-gateway"]`
  to the `api` service in Dograh's `docker-compose.yaml` and use `http://host.docker.internal:8890/v1`.
- **Dograh on another machine:** the GPU machine's LAN or VPN (e.g. Tailscale) address.
- **Dograh Cloud:** the URL must be public HTTPS (put the server behind a reverse proxy or tunnel, and set `--api-key`).

Dograh asks for raw 16-bit 24 kHz mono PCM, which is what Hurston-1 produces, so there is no resampling in between.
On phone calls Dograh converts it for the carrier.

## Voices

`GET /v1/audio/voices` lists them. Each accent comes as `-female`, `-male`, `-female-older` and `-male-older`:

| Voice name prefix | Accent |
|---|---|
| `southern` | Southern American |
| `american` | American (General) |
| `nigerian` | Nigerian English |
| `yoruba` | Nigerian (Yoruba) English |
| `igbo` | Nigerian (Igbo) English |
| `west-african` | West African English |
| `east-african` | East African English |
| `swahili` | East African (Swahili) English |
| `southern-african` | Southern African English |

Instead of a name you can put a full Hurston description in the Voice field, in the order the model was trained on
(gender, age, accent, style, pace):

```
Male, middle-aged, Nigerian (Igbo) English accent, clear conversational voice, measured pacing
```

**Speed** picks one of the trained pace labels instead of time-stretching the audio: below 0.8 is slow, 0.8-0.95
measured, 0.95-1.15 natural, above 1.15 fast.

## Know before you ship

- **The voice can change between sentences.** Hurston-1 makes a fresh voice for every request that fits the
  description, and Dograh sends each sentence of the agent's reply as its own request. Within one call you will hear
  the same accent and gender, but not always the same person. Consistent voices are planned for a later release.
- Names, dates and amounts are the weakest content; test your prompts and listen.
- The `clear conversational voice` style is the most intelligible. All preset voices use it. A
  `casual conversational voice` in a custom description is misheard noticeably more often.
- English only, no child voices. Every voice is synthetic; there is no voice cloning.

## Good fits in Dograh

- **Test callers.** Point a Dograh agent playing a caller (customer, patient, applicant) at your production agent and
  hear how it handles accents before real callers do.
- **Accent-coverage demos and prototypes** of agents for communities in the US South and across Africa.
- **Short, one-sentence prompts** (confirmations, reminders, menus), where the sentence-to-sentence voice change does
  not come up.
