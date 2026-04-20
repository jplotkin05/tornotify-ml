# tornotify-ml

Radar ML scanner for live NEXRAD Level-II tornado-signature scoring with TorNet.

## Inspiration
Back on September 24, 2001, a devastating tornado crossed over areas of North Campus. Two students were killed, 57 were injured, 300 cars were destroyed, and dorms and trees were damaged. Since then, the University has used tornado sirens to improve early warning communications. Tornado detection remains difficult for forecasters at weather offices.

MIT Lincoln Laboratory published a benchmark dataset for detecting tornadic signatures in weather radar data. We use that model foundation with realtime radar data to pursue earlier tornado detection and give people more time to take shelter.

## What it does
The model ingests realtime and archived Next Generation Weather Radar data (NEXRAD) served from AWS S3. The latest frame from each radar site is filtered for high reflectivity to find storm cells. Present cells are cropped, extrapolated, and passed into tornado classification. Processed cells are plotted on a map with tornado probabilities.

To reduce anomalous predictions, the program aggregates future radar time steps for consistency. If a tornado remains likely, its path is plotted on the map.

The dashboard can also review past weather events dating back to 2013.

## How we built it
The system is written in Python. It loads the MIT model from Hugging Face and connects it to live data ingestion built with `nexradaws`. Radar images with state and county borders are generated with Py-ART and Cartopy. The dashboard is built with Streamlit.

## Challenges we ran into
Single frames can return false positive tornado signatures, so persistent detections must be validated over multiple radar time steps. If the next time step shows a similar tornado probability in the same region following computed storm-track movement, the system elevates the detection instead of treating it as an anomaly.

## Accomplishments that we're proud of
Training data spans 2013 through 2022. We wanted to test whether the system could identify tornado events beyond the training set. A key result was passing archived radar data from the 2023 Rolling Fork tornado in Mississippi and getting an accurate classification and storm path. Other recent tornadoes outside the dataset showed similar results.

## What we learned
We learned how to interpret radar data and the six components of modern radar: reflectivity, radial velocity, spectrum width, differential reflectivity, correlation coefficient, and specific differential phase. That helped us understand what the model processes and how to interpret performance across severe weather examples.

## What's next for Tornotify - Storm Early Warning Systems
We want to go beyond standard mobile phone warning systems and integrate detection into smart home devices, including stereo warnings, lights changing colors, and other features that can warn people away from their phones and people with sensory disabilities.

## Scanner Modes

### Local adaptive scanner

Runs in one Python process with a thread pool:

```bash
python tornotify/scripts/run_pipeline.py --all-sites --loop --workers 4 --no-images
```

This is simplest for local testing, but processed scan state and temporal tracks live in that one process.

### Redis distributed scanner

Runs multiple worker processes against a Redis-backed queue. Redis stores:

- due-time priority queue for radar sites
- per-site processing locks
- processed scan keys
- temporal track state

CSV remains the output format:

- `tornotify/data/results.csv` for every scored storm cell
- `tornotify/data/detections.csv` for temporally confirmed tracks

Install the Redis client dependency with the normal requirements:

```bash
pip install -r tornotify/requirements.txt
```

Start Redis separately, for example:

```bash
redis-server
```

One-machine launcher:

```bash
python tornotify/scripts/run_distributed.py --all-sites --workers 12 --requeue-now
```

Separate scheduler and workers:

```bash
python tornotify/scripts/distributed_scheduler.py --all-sites --loop
python tornotify/scripts/distributed_worker.py --workers 12
```

Workers default to no radar PNG output for throughput. Add `--image-dir tornotify/data/radar_images` when image artifacts are needed.

## Useful Settings

```bash
export REDIS_URL=redis://localhost:6379/0
export TORNOTIFY_DISTRIBUTED_WORKERS=12
export TORNOTIFY_REDIS_PREFIX=tornotify
export TORNOTIFY_DISTRIBUTED_SITE_LOCK_TTL_SECONDS=900
export TORNOTIFY_DISTRIBUTED_PROCESSED_TTL_SECONDS=259200
export TORNOTIFY_DISTRIBUTED_TRACK_TTL_SECONDS=21600
```

Polling priority still uses the scanner timing settings:

```bash
export TORNOTIFY_SCANNER_HOT_POLL_SECONDS=60
export TORNOTIFY_SCANNER_ACTIVE_POLL_SECONDS=120
export TORNOTIFY_SCANNER_QUIET_POLL_SECONDS=300
export TORNOTIFY_SCANNER_NO_SCAN_POLL_SECONDS=600
```

## Notes

- Use `--all-sites` for the full non-TDWR radar catalog.
- Use `--sites KMAF KTWX KILX` for a targeted scan group.
- Multiple worker hosts can share the same Redis queue.
- Multiple local worker processes append to the same CSV files using sidecar file locks.
- On a 24-core server, start around `--workers 8` to `--workers 12`, then raise only if CPU, memory, model load time, and S3 download rate stay healthy.
