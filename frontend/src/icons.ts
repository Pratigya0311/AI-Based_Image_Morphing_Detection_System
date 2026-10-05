export type IconName = 'shield' | 'upload' | 'image' | 'check' | 'arrow' | 'close' | 'files' | 'mark'

const icons: Record<IconName, string> = {
  shield: '<svg viewBox="0 0 24 24"><path d="M12 3 20 6.5v5.8c0 4.8-3.2 8.2-8 9.7-4.8-1.5-8-4.9-8-9.7V6.5L12 3Z"/><path d="m8.5 12.2 2.2 2.2 4.8-5"/></svg>',
  upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/></svg>',
  image: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m4 18 5-5 3.5 3.5 2.5-2.5 5 4"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="m5 12.5 4.2 4.2L19 7"/></svg>',
  arrow: '<svg viewBox="0 0 24 24"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>',
  close: '<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>',
  files: '<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6M8 14h8M8 17h6"/></svg>',
  mark: '<svg viewBox="0 0 32 32"><path d="M5.5 16c2.8-4.6 6.3-6.9 10.5-6.9S23.7 11.4 26.5 16c-2.8 4.6-6.3 6.9-10.5 6.9S8.3 20.6 5.5 16Z"/><circle cx="16" cy="16" r="3.25"/><path d="M16 4.5v2.2M16 25.3v2.2M4.5 16h2.2M25.3 16h2.2"/></svg>',
}

export const icon = (name: IconName) => icons[name]
