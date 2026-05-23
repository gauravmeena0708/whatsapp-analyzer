## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.
## 2025-05-23 - Async Action Feedback
**Learning:** Adding explicit visual feedback (disabling buttons, aria-busy states, text changes) during potentially long-running async JavaScript operations (like local file parsing) is critical to prevent user confusion and double-submissions.
**Action:** Always ensure async trigger buttons are disabled and visually indicate a loading state in Vanilla JS projects where React/framework state management isn't present.
