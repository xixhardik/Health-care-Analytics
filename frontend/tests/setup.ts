import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// Next's router is not available in jsdom, so the pieces components actually use
// are stubbed. Anything beyond push/replace would be unused indirection.
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
  }),
  usePathname: () => "/",
  useParams: () => ({ id: "abc123abc123abcd" }),
}));

/*
 * jsdom has no layout engine, so every element measures 0x0. Recharts'
 * ResponsiveContainer reads `getBoundingClientRect()` on mount and warns
 * "The width(0) and height(0) of chart should be greater than 0" when it gets
 * nothing back.
 *
 * The fix is to supply the measurements jsdom cannot compute, not to silence the
 * warning: the box below is a plausible panel size, and the charts then lay out
 * for real. Production chart code is untouched — in a browser the same call
 * returns the browser's own measurement.
 */
const TEST_VIEWPORT = { width: 640, height: 320 } as const;

Object.defineProperty(window.HTMLElement.prototype, "offsetWidth", {
  configurable: true,
  value: TEST_VIEWPORT.width,
});
Object.defineProperty(window.HTMLElement.prototype, "offsetHeight", {
  configurable: true,
  value: TEST_VIEWPORT.height,
});

window.Element.prototype.getBoundingClientRect = function getBoundingClientRect(): DOMRect {
  const { width, height } = TEST_VIEWPORT;
  return {
    width,
    height,
    top: 0,
    left: 0,
    right: width,
    bottom: height,
    x: 0,
    y: 0,
    toJSON: () => ({ width, height, top: 0, left: 0, right: width, bottom: height, x: 0, y: 0 }),
  };
};

// jsdom ships no ResizeObserver. Nothing resizes in a test, so observing is a
// no-op; the initial size above is what the charts lay out against.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver =
  globalThis.ResizeObserver ?? (ResizeObserverStub as never);
