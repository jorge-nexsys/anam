import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "AnamDB",
  description: "The AI-Native Neurosymbolic Database Engine",
  base: "/anam/", // For GitHub Pages deployment
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/what-is-anamdb' },
      { text: 'Hub', link: '/registry/index.json' }
    ],
    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'What is AnamDB?', link: '/guide/what-is-anamdb' },
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Core Concepts', link: '/guide/core-concepts' }
        ]
      },
      {
        text: 'Developer Guide',
        items: [
          { text: 'CLI & REPL Reference', link: '/guide/cli-repl' },
          { text: 'Rust SDK Integration', link: '/guide/rust-sdk' },
          { text: 'Logic Packs & Hub', link: '/guide/logic-packs-hub' }
        ]
      },
      {
        text: 'Advanced Features',
        items: [
          { text: 'Explainability & HITL', link: '/guide/explainability-hitl' },
          { text: 'Distributed Reasoning Plane', link: '/guide/distributed' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/jorge-nexsys/anam' }
    ],
    footer: {
      message: 'Released under the BSL-1.1 License.',
      copyright: 'Copyright © 2026 Jorge Martinez'
    }
  }
})
