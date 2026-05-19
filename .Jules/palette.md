## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2025-02-13 - Explicit Loading States for Async Buttons
**Learning:** Form submit buttons associated with async tasks (like file parsing) must explicitly disable themselves and show a progress state, even if a global loading indicator is present. Leaving the button active allows multiple unintended submissions and confuses users about whether their action registered.
**Action:** Always add a disabled state, `aria-busy="true"`, and progress text (e.g., "Analyzing...") to primary action buttons during asynchronous operations, restoring them once complete.
