## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-16 - ARIA Roles for Canvas Chart Elements
**Learning:** `<canvas>` elements used for data visualization (like Chart.js charts) are entirely opaque to screen readers by default. They are treated as empty elements without semantic meaning.
**Action:** Always add `role="img"` and a descriptive `aria-label` (e.g., `aria-label="Messages per User Chart"`) to all `<canvas>` elements to ensure data visualizations are minimally accessible and their purpose is announced.
