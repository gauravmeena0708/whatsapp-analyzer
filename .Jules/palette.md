## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-16 - ARIA roles for Canvas elements
**Learning:** `<canvas>` elements are inherently inaccessible to screen readers because they just render pixels. They need proper ARIA roles to provide context.
**Action:** Always add `role="img"` and a descriptive `aria-label` to all `<canvas>` elements (like charts) so screen readers can announce them correctly and describe what they display.
