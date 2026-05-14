## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2026-05-14 - Accessible HTML Canvas Elements for Charts
**Learning:** `<canvas>` elements used for rendering visual charts (like Chart.js) are opaque to screen readers by default. To make them accessible, they need to be assigned an explicit `role="img"` along with a descriptive `aria-label`.
**Action:** When creating or dynamically rendering `<canvas>` charts, always provide `role="img"` and an appropriate `aria-label` attribute (e.g., extracting the chart title dynamically from the configuration if generated via a Python backend, or hardcoding it for static templates).
