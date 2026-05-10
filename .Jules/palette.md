## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2026-05-10 - ARIA Roles and Labels for Canvas Charts
**Learning:** `<canvas>` elements used for rendering charts are inherently inaccessible to screen readers because they do not provide semantic information or alternative text natively.
**Action:** Always add `role="img"` and a descriptive `aria-label` (e.g., `aria-label="Hourly Activity Chart"`) to `<canvas>` elements to ensure screen readers can announce them as images and describe their content or purpose.
