## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-15 - ARIA Roles for Canvas Elements
**Learning:** HTML `<canvas>` elements are completely opaque to screen readers by default. Screen reader users will not know what the canvas represents or even that it exists.
**Action:** Always add `role="img"` and a descriptive `aria-label` attribute to `<canvas>` elements that display charts or visual information, ensuring screen readers can announce them as images with a specific purpose.
