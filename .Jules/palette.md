## 2024-05-15 - ARIA Roles for Custom Spinners
**Learning:** Custom CSS spinners implemented as `<div>` elements are invisible to screen readers without ARIA roles. Adding `role="status"` and `aria-label` provides necessary context.
**Action:** Always add `role="status"` and `aria-label="Loading..."` to custom CSS-only loading indicators across all views.

## 2024-05-17 - ARIA Roles for Canvas Chart Elements
**Learning:** `<canvas>` elements used for charts are completely opaque to screen readers by default. Since they are used extensively to display critical data insights in this app, leaving them without accessible labels prevents visually impaired users from knowing what data is being presented.
**Action:** Always add `role="img"` and a descriptive `aria-label` to all `<canvas>` elements, both in static HTML templates and dynamically generated strings, to provide meaningful context.
