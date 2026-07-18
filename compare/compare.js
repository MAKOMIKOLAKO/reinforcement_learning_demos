/**
 * compare/compare.js
 *
 * Enhances the plain `.glossary-term` spans in compare/index.html into full
 * accessible tooltips using the shared glossary component.
 *
 * ASSUMED API (shared/glossary.js is being added in a parallel PR and was not
 * yet present in this branch's shared/ directory at the time this was
 * written): a named export `glossaryTerm(term)` that returns an HTML string
 * for an accessible tooltip-enhanced span covering `term`, working on hover,
 * click/tap, and keyboard focus. Confirm this signature once shared/glossary.js
 * lands and adjust the call below if it differs (e.g. a different export
 * name, or a signature like `glossaryTerm(term, displayText)` /
 * `attachGlossaryTooltip(element, term)`).
 *
 * If shared/glossary.js isn't present yet, or its API doesn't match this
 * assumption, this script fails quietly and the page still works: the
 * `.glossary-term` spans just render as plain text with the dotted-underline
 * affordance defined in compare.css instead of a full tooltip.
 */

async function enhanceGlossaryTerms() {
  let glossaryTerm;
  try {
    ({ glossaryTerm } = await import("../shared/glossary.js"));
  } catch (err) {
    console.warn(
      "compare.js: shared/glossary.js not available yet, leaving glossary terms as plain text.",
      err
    );
    return;
  }

  if (typeof glossaryTerm !== "function") {
    console.warn(
      "compare.js: shared/glossary.js did not export a `glossaryTerm` function; " +
        "leaving glossary terms as plain text. Check the actual export name/signature."
    );
    return;
  }

  document.querySelectorAll(".glossary-term[data-glossary]").forEach((el) => {
    const term = el.dataset.glossary;
    try {
      const html = glossaryTerm(term, el.textContent);
      if (typeof html === "string" && html.trim()) {
        el.outerHTML = html;
      }
    } catch (err) {
      console.warn(`compare.js: glossaryTerm("${term}") failed, leaving plain text.`, err);
    }
  });
}

enhanceGlossaryTerms();
