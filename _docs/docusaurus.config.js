// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
// Ref: https://github.com/infinum/eightshift-docs/blob/develop/website/docusaurus.config.js

const math = require('remark-math');
const katex = require('rehype-katex');
// const oembed = require('@agentofuser/remark-oembed')
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI',
  tagline: 'AI and ML utils',
  url: 'https://docs.recohut.com',
  baseUrl: '/recohut/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/logo.svg',
  organizationName: 'sparsh-ai',
  projectName: 'ai',
  plugins: [require.resolve("@cmfcmf/docusaurus-search-local")],
  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
          docs: {
              sidebarPath: require.resolve('./sidebars.js'),
              editUrl: 'https://github.com/sparsh-ai/recohut',
              remarkPlugins: [math],
              rehypePlugins: [katex],
            //   lastVersion: 'current',
            //   versions: {
            //       current: {
            //       label: '1.0.0',
            //       path: '1.0.0',
            //       },
            //   },
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/sparsh-ai/recohut/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
      ],
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Recohut',
        logo: {
          alt: 'Recohut Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Docs',
          },
          {to: '/blog', label: 'Blog', position: 'left'},
        //   {
        //     type: 'docsVersionDropdown',
        //     position: 'right',
        //     dropdownItemsAfter: [{to: '/versions', label: 'All versions'}],
        //     dropdownActiveClassDisabled: true,
        //   },
          {
            href: 'https://github.com/sparsh-ai/recohut',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Learn',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
                },
              {
                label: 'Concepts',
                to: '/docs/concept-basics',
                },
              {
                label: 'Tutorials',
                to: '/docs/tutorials',
                },
              {
                label: 'Projects',
                to: '/docs/projects',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/sparsh-ai/recohut',
              },
              {
                label: 'Jupyter Notebooks',
                href: 'https://nb.recohut.com/',
              },
              {
                label: 'Interactive Stories',
                href: 'https://step.recohut.com/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Recohut Docs, Inc. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;