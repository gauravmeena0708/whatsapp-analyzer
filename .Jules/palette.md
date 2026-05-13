## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-16 - ARIA Roles for HTML Canvas Elements
**Learning:** Raw `<canvas>` elements used for charts (like Chart.js) are opaque to screen readers. They act like images but don't inherently have alternative text semantics.
**Action:** Always add `role="img"` and a descriptive `aria-label` (e.g. the chart title) to `<canvas>` elements to ensure screen readers can announce the presence and purpose of the chart.
