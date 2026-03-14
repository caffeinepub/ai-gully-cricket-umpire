# AI Gully Cricket Umpire — Version 3

## Current State
A browser-based, frontend-only React + TypeScript app using TensorFlow.js MoveNet Thunder for real-time cricket bowling knee angle analysis. Features: live video with skeleton overlay, full-body detection gate, stability filter, delivery lock at minimum angle, camera swap (back/front), locked verdict display with timestamp.

## Requested Changes (Diff)

### Add
- **Ball Release Detection**: Enhance existing delivery lock with clearer "ball release" moment detection — already based on angle inflection; add release flash animation and explicit release moment marker in UI.
- **Sound Effects**: Web Audio API synthesized sounds — siren waveform for NO BALL, short click/beep for LEGAL BALL. Play on every locked delivery.
- **Delivery Counter**: Persistent session stats panel showing Total Balls, Legal Balls, No Balls. Increments on each locked delivery.
- **Replay System**: Capture canvas frames into a circular buffer during live detection. On delivery lock, show a slow-motion frame-by-frame replay overlay displaying the 40 frames around the release moment.
- **Video Recording**: Use MediaRecorder on the camera stream to record each delivery. After lock, save the clip as a Blob; offer Download and Replay buttons in the decision banner.
- **Crease Line Detection**: User-adjustable virtual crease line overlay on the video canvas. User drags a horizontal line to match the bowling crease. Check if ankle keypoint (y-coordinate) crosses the line — flag a separate "Crease Overstepped" warning.
- **Decision Confidence Score**: Compute confidence % from average keypoint scores of hip/knee/ankle + stability buffer fullness. Show as "XX% confidence" badge on the locked decision.
- **Tournament Mode**: A modal/sheet setup screen (accessible via a "Tournament Mode" button) with fields: Team A name, Team B name, current bowler name. Stats panel shows per-bowler no-ball count. Tournament header appears in main UI when active.
- **Offline AI Model**: Add note in UI that model is cached after first load. Use TF.js model caching via IndexedDB by passing `modelUrl` with `tf.io.withFile` or just preserve CDN with service worker cache. For now, add a "Model Cached" indicator after first successful load using localStorage flag.

### Modify
- Controls area: add Camera Swap button more prominently, add Tournament Mode button.
- Decision banner: add confidence score, crease warning, download/replay buttons for video clip.
- Status bar: show offline model cache status.
- How It Works section: update to reflect new features.

### Remove
- Nothing removed.

## Implementation Plan
1. Add session stats state (totalBalls, legalBalls, noBalls) — increment on lockedVerdict.
2. Web Audio API sound synthesis — siren for NO BALL (oscillator 440→880Hz sweep), click for LEGAL BALL (short noise burst).
3. Frame buffer: store canvas ImageData snapshots (or dataURLs at reduced size) in a rolling 60-frame buffer. On delivery, extract ±20 frames around lock moment.
4. Replay modal: show extracted frames animating at reduced speed (150ms/frame = ~7fps = slow-mo from 30fps).
5. MediaRecorder: start recording when camera starts, on each delivery lock grab the last ~3 seconds as a Blob chunk. Offer download link and inline video replay.
6. Crease line: SVG or canvas overlay with a draggable horizontal line. Default at 60% of frame height. Check ankle Y vs line Y each frame.
7. Confidence score: mean of hip/knee/ankle scores × stability buffer fill ratio × 100.
8. Tournament Mode modal: form with team names + bowler. Display as header chip when active. Track per-bowler stats in a map.
9. Model cache indicator: after model loads, set localStorage key; show green "Cached" chip.
10. Delivery counter panel below video.
