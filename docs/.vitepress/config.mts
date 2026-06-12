import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "AnamDB",
  description: "The AI-Native Neurosymbolic Database Engine",
  base: "/anam-db/", // GitHub Pages: anamdb.github.io/anam-db
  themeConfig: {
    logo: {
      light: '/full_logo_dark.png',
      dark: '/transparent_full_logo.png'
    },
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Engine Guide', link: '/guide/what-is-anamdb' },
      { text: 'CLI Reference', link: '/cli/getting-started' }
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'What is AnamDB?', link: '/guide/what-is-anamdb' },
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'SDK Integrations', link: '/guide/rust-sdk' }
          ]
        },
        {
          text: 'Concepts & Engine',
          items: [
            { text: 'Core Concepts', link: '/guide/core-concepts' },
            { text: 'Datalog & Logic Packs', link: '/guide/logic-packs-hub' },
            { text: 'Model Optimization & Pareto', link: '/guide/model-optimization' },
            { text: 'Hybrid Storage & Vector', link: '/guide/storage-lance' },
            { text: 'Explainability & HITL', link: '/guide/explainability-hitl' }
          ]
        },
        {
          text: 'Platform & Operations',
          items: [
            { text: 'Distributed Reasoning Plane', link: '/guide/distributed' },
            { text: 'Security & Secrets', link: '/guide/security' },
            { text: 'Limits & Benchmarks', link: '/guide/limits' }
          ]
        }
      ],
      '/cli/': [
        {
          text: 'Command Line Interface',
          items: [
            { text: 'Getting Started', link: '/cli/getting-started' },
            { text: 'Interactive REPL', link: '/cli/repl' }
          ]
        },
        {
          text: 'CLI Commands',
          items: [
            { text: 'init', link: '/cli/commands/init' },
            { text: 'start', link: '/cli/commands/start' },
            { text: 'serve', link: '/cli/commands/serve' },
            { text: 'status', link: '/cli/commands/status' }
          ]
        },
        {
          text: 'REPL Dot-Commands',
          items: [
            { text: '.load & .ingest', link: '/cli/repl/load-ingest' },
            { text: '.logic & .rules', link: '/cli/repl/logic-rules' },
            { text: '.model & .models', link: '/cli/repl/models' },
            { text: '.explain', link: '/cli/repl/explain' },
            { text: '.hub', link: '/cli/repl/hub' },
            { text: '.devices', link: '/cli/repl/devices' }
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AnamDB/anam-db' }
    ],
    footer: {
      message: 'Released under the Apache 2.0 License.',
      copyright: 'Copyright © 2026 AnamDB'
    }
  }
})
