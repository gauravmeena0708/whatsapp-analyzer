## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2026-05-21 - Asynchronous Button States
**Learning:** Primary action buttons for asynchronous operations in HTML templates must be disabled, have `aria-busy="true"`, and show progress text (e.g., 'Analyzing...') during execution to prevent duplicate submissions and enhance accessibility.
**Action:** Always add `aria-busy="true"`, `disabled=true`, and progress text to primary action buttons across all views when executing operations that block the thread or wait for a response.
