## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2024-05-20 - Communicating Async State on Inputs
**Learning:** While buttons trigger async operations and often receive `disabled` and `aria-busy` states, associated file inputs (like `chatFileInput`) are often overlooked. Leaving inputs enabled during async reading can allow unintended re-triggers. Adding `disabled` and `aria-busy="true"` to both the input and associated UI elements (like primary buttons or dropdowns) ensures the full interactive surface is locked down and communicates progress to assistive tech.
**Action:** Always disable and set `aria-busy="true"` on file inputs alongside their submit buttons during asynchronous read operations, ensuring states are reverted in `finally` or `onerror` blocks.

## 2024-05-19 - Added title tooltips to disabled interactive elements
**Learning:** Users often do not know why a form element, like a button or a select dropdown, is disabled on initial load. Adding a native HTML `title` attribute to disabled interactive elements explicitly explains the prerequisite actions required (e.g., "Upload a chat file to enable analysis"), improving both accessibility and UX clarity.
**Action:** Always add descriptive `title` attributes to disabled interactive elements to explain their state, and use JavaScript to toggle or remove them when the element's state changes (e.g., when a file is uploaded or processing begins). Also, clearly indicate inactive states visually using `disabled:opacity-50` for file inputs and selects, matching the existing button styles.
