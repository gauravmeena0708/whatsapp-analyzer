## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2024-05-20 - Communicating Async State on Inputs
**Learning:** While buttons trigger async operations and often receive `disabled` and `aria-busy` states, associated file inputs (like `chatFileInput`) are often overlooked. Leaving inputs enabled during async reading can allow unintended re-triggers. Adding `disabled` and `aria-busy="true"` to both the input and associated UI elements (like primary buttons or dropdowns) ensures the full interactive surface is locked down and communicates progress to assistive tech.
**Action:** Always disable and set `aria-busy="true"` on file inputs alongside their submit buttons during asynchronous read operations, ensuring states are reverted in `finally` or `onerror` blocks.

## 2024-05-22 - Explaining Disabled States with Tooltips
**Learning:** When interactive elements like `<button>` or `<select>` are initially disabled or become disabled during an error state, users are often confused about *why* they cannot interact with them. Simply greying them out is not enough. Adding a native HTML `title` attribute to disabled elements provides a tooltip that explains the required precondition (e.g., "Please select a chat file to analyze" or "Please upload a chat file first"). Furthermore, it's critical to ensure that error handlers (like `FileReader.onerror`) don't incorrectly re-enable these elements if the precondition wasn't actually met.
**Action:** Always add descriptive `title` attributes to disabled interactive elements. Toggle these titles using JavaScript when the element's state changes. Ensure error handling blocks maintain the disabled state if the required action (like successfully reading a file) failed.
