## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2024-05-20 - Communicating Async State on Inputs
**Learning:** While buttons trigger async operations and often receive `disabled` and `aria-busy` states, associated file inputs (like `chatFileInput`) are often overlooked. Leaving inputs enabled during async reading can allow unintended re-triggers. Adding `disabled` and `aria-busy="true"` to both the input and associated UI elements (like primary buttons or dropdowns) ensures the full interactive surface is locked down and communicates progress to assistive tech.
**Action:** Always disable and set `aria-busy="true"` on file inputs alongside their submit buttons during asynchronous read operations, ensuring states are reverted in `finally` or `onerror` blocks.

## 2024-05-24 - Provide Explicit Titles for Disabled State Interactions
**Learning:** Screen readers and keyboard users often struggle to understand *why* an interactive element (like a button or select dropdown) is disabled. In this app, relying solely on `disabled` or `cursor-not-allowed` styles leaves users guessing when they can interact. Additionally, error handlers (like `FileReader.onerror` or catch blocks) sometimes incorrectly re-enable interactive elements when their required precondition (e.g., successful data loading) has failed, creating dead ends.
**Action:** Add native HTML `title` attributes to disabled interactive elements to explicitly explain the precondition required to enable them (e.g., `title="Upload a chat file first"`). Dynamically update this `title` via JavaScript during asynchronous processing (`title="Analyzing..."`) and remove it when the element becomes active. Ensure that error handling blocks explicitly re-disable or maintain the disabled state of interactive elements if the data they require is missing.
