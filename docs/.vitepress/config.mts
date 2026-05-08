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
          { text: 'Getting Started', link: '/guide/getting-started' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/jam5991/anam' }
    ],
    footer: {
      message: 'Released under the BSL-1.1 License.',
      copyright: 'Copyright © 2026 Jorge Martinez'
    }
  }
})
