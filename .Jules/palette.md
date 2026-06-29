## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-18 - Canvas Accessibility for Screen Readers
**Learning:** <canvas> elements are opaque to screen readers by default. Adding `role="img"` and an `aria-label` ensures visually impaired users are aware that a chart or data visualization is present on the page.
**Action:** Always add `role="img"` and descriptive `aria-label` to all `<canvas>` elements across HTML templates and dynamically generated plots.

## 2024-05-20 - Communicating Async State on Inputs
**Learning:** While buttons trigger async operations and often receive `disabled` and `aria-busy` states, associated file inputs (like `chatFileInput`) are often overlooked. Leaving inputs enabled during async reading can allow unintended re-triggers. Adding `disabled` and `aria-busy="true"` to both the input and associated UI elements (like primary buttons or dropdowns) ensures the full interactive surface is locked down and communicates progress to assistive tech.
**Action:** Always disable and set `aria-busy="true"` on file inputs alongside their submit buttons during asynchronous read operations, ensuring states are reverted in `finally` or `onerror` blocks.

## 2024-07-25 - Prevent Incorrect Re-enabling of Dependent Controls
**Learning:** In asynchronous flows (like reading files), if a process fails in a `catch` or `onerror` block, dependent UI elements (like submit buttons or next-step dropdowns) should not be blindly re-enabled. Additionally, relying solely on visual disabled states (like opacity or cursor-not-allowed) isn't enough; native HTML `title` attributes provide crucial context to screen readers and mouse users about *why* the element is disabled (e.g., "Please select a file first" or "Error processing file").
**Action:** Always verify that preconditions are met before re-enabling elements in error handling blocks, and always pair visual disabled states with semantic HTML `title` attributes explaining the state.
