## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2024-05-20 - Communicating Async State on Inputs
**Learning:** While buttons trigger async operations and often receive `disabled` and `aria-busy` states, associated file inputs (like `chatFileInput`) are often overlooked. Leaving inputs enabled during async reading can allow unintended re-triggers. Adding `disabled` and `aria-busy="true"` to both the input and associated UI elements (like primary buttons or dropdowns) ensures the full interactive surface is locked down and communicates progress to assistive tech.
**Action:** Always disable and set `aria-busy="true"` on file inputs alongside their submit buttons during asynchronous read operations, ensuring states are reverted in `finally` or `onerror` blocks.
## 2024-06-10 - Explicit Disabled States and Error Recovery
**Learning:** Users often don't know *why* an input or button is disabled. In `visual_2.html` and `visual_3.html`, a select dropdown was blindly re-enabled during `FileReader.onerror` and `catch` blocks even if the chat file wasn't successfully parsed. This left users with an active but empty dropdown, causing confusion.
**Action:** Always add native HTML `title` attributes (or similar tooltips/aria-descriptions) to disabled interactive elements to explicitly explain the required precondition (e.g., `title="Please upload and parse a chat file first"`). Ensure elements are only re-enabled when their specific preconditions are fully met, not generically during `finally` or `onerror` blocks.
