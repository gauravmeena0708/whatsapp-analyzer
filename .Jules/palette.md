## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2024-05-20 - Communicating Async State on Inputs
**Learning:** While buttons trigger async operations and often receive `disabled` and `aria-busy` states, associated file inputs (like `chatFileInput`) are often overlooked. Leaving inputs enabled during async reading can allow unintended re-triggers. Adding `disabled` and `aria-busy="true"` to both the input and associated UI elements (like primary buttons or dropdowns) ensures the full interactive surface is locked down and communicates progress to assistive tech.
**Action:** Always disable and set `aria-busy="true"` on file inputs alongside their submit buttons during asynchronous read operations, ensuring states are reverted in `finally` or `onerror` blocks.

## 2024-05-22 - Explaining Disabled States with Native Tooltips
**Learning:** Users can be confused when interactive elements (like buttons or dropdowns) are disabled by default without clear explanation. Using native HTML `title` attributes on disabled elements provides an accessible, zero-dependency tooltip explaining *why* the element is disabled (e.g., "Please upload a file first"). These titles must be dynamically removed when the element is enabled, and explicitly updated during error states to prevent confusing residual messages. Additionally, error handlers (like `FileReader.onerror`) must be careful not to re-enable elements whose preconditions (like successfully reading a file) were not met.
**Action:** Always add native `title` attributes to disabled interactive elements to explain the required action, and toggle/update these attributes dynamically in JavaScript as the element's state changes. Never re-enable an element in an error handler if its core dependency failed.
