// 依存を増やさないための、最小のインラインSVGアイコン(装飾なので aria-hidden)。
import type { ReactNode } from "react";

function Svg({ children, size = 18 }: { children: ReactNode; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export const IconMoon = () => (
  <Svg>
    <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />
  </Svg>
);
export const IconFork = () => (
  <Svg>
    <path d="M7 3v8M4 3v5a3 3 0 0 0 6 0V3M7 11v10M17 3c-2 1.5-3 4-3 7v3h3v8" />
  </Svg>
);
export const IconBreakfast = () => (
  <Svg>
    <path d="M4 11h13v3a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5v-3ZM17 12h1.5a2.5 2.5 0 0 1 0 5H16M8 3c0 1.5 1.5 1.5 1.5 3M12 3c0 1.5 1.5 1.5 1.5 3" />
  </Svg>
);
export const IconHeart = () => (
  <Svg>
    <path d="M3 12h4l2-5 4 10 2-5h6" />
  </Svg>
);
export const IconSend = () => (
  <Svg size={20}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </Svg>
);
export const IconSpark = () => (
  <Svg size={16}>
    <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
  </Svg>
);
export const IconUser = () => (
  <Svg size={16}>
    <circle cx="12" cy="8" r="3.5" />
    <path d="M5 20c1-4 4-5.5 7-5.5s6 1.5 7 5.5" />
  </Svg>
);
export const IconInfo = () => (
  <Svg size={15}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8h.01" />
  </Svg>
);
export const IconTool = () => (
  <Svg size={15}>
    <path d="M14.7 6.3a4 4 0 0 0-5 5L3 18l3 3 6.7-6.7a4 4 0 0 0 5-5l-2.4 2.4-2.6-.6-.6-2.6 2.6-2.2Z" />
  </Svg>
);

export const SUGGESTION_ICONS = [IconMoon, IconFork, IconBreakfast, IconHeart];

/** ロゴマーク(グラデーションの角丸+脈拍) */
export function LogoMark() {
  return (
    <span className="logo-mark" aria-hidden="true">
      <IconHeart />
    </span>
  );
}
