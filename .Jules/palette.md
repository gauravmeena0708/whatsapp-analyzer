## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-15 - ARIA Roles for Canvas Charts
**Learning:** `<canvas>` elements used for rendering data visualizations are treated as empty, meaningless elements by screen readers. They provide no context about the chart they display. Adding `role="img"` and a descriptive `aria-label` ensures screen reader users understand what the visual element represents.
**Action:** Always add `role="img"` and a concise, descriptive `aria-label` to all `<canvas>` elements displaying charts or visual data across all applications and templates.
