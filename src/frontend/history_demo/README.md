# Saved-video workspace

The canonical HTML entry is `../online_web/history-demo/index.html`. Vite bundles the modules in this directory into the same `online_web/dist/` as the capture and independent memory pages.

The workspace lists ready sessions, plays their bound videos, shows memory records, asks ordinary questions and displays timestamped visual evidence. Playback position does not restrict retrieval: questions use the selected session's already generated memories, with any explicit time requirement handled by the backend.

Uploads persist a receipt before sending. The UI polls the original task after a refresh, exposes failures and resume controls, and permits replay/Q&A only after the backend reports both published memory and ready retrieval indexes. It never automatically repeats an uncertain upload or question submission. Memory edits retain drafts and use the backend's version/idempotency contract.

Language/theme/font preferences are shared with the React pages. User questions, answers, memory content and session names remain verbatim when the interface language changes. Images use the page origin or the explicitly configured backend origin; saved descriptions are rendered as text.

Run `npm test` here for Node controller tests. Run `npm run test:browser` from `../online_web` for a production build plus offline Chromium checks. Browser setup and deployment details are documented in `../README_DEPLOY.md`.
