## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-17 - Accessible Chart.js Canvas Elements
**Learning:** Chart.js `<canvas>` elements are rendered as images but are not inherently accessible to screen readers, causing them to be ignored or poorly described.
**Action:** Always add `role="img"` and a descriptive `aria-label` to dynamically rendered `<canvas>` elements so screen readers can announce them as data visualizations.
