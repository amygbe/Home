# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal portfolio/blog website built with Hugo (static site generator). Uses the Soho theme with extensive customizations.

- **Site URL**: https://www.amybell.info/
- **Author**: Amy Bell (Data Scientist)

## Build Commands

```bash
# Local development build
hugo

# Production build (used by CI/CD)
hugo --gc --minify

# Start local dev server (default port 1313)
hugo server
```

**Requirements**: Hugo v0.128.0+ (extended version), Dart Sass

## Project Architecture

### Content Structure
- `content/` - Markdown files with TOML front matter
  - `about/`, `contact/`, `homepage/` - Static pages
  - `posts/` - Blog posts
- `archetypes/default.md` - Template for new content

### Customization Layers
- `layouts/` - Custom Hugo templates overriding theme
- `layouts/shortcodes/` - Custom components (contact form, project cards, bookshelf, tier list, gooey buttons)
- `static/css/custom.css` - Site-wide styling (630 lines)
- `static/css/syntax.css` - Code syntax highlighting

### Theme
Using `themes/soho/` (the `ananke` theme is present but not active)

### Build Output
`public/` - Generated static site (committed to repo)

## Deployment

Automatic via GitHub Actions (`.github/workflows/hugo.yaml`):
- Triggers on push to `main` branch
- Builds and deploys to GitHub Pages

## Design System

- **Primary color**: Green (#2f523e)
- **Secondary color**: Wheat (#F5DEB3)
- Contact form uses Formspree integration

## Adding Content

New posts require TOML front matter:
```toml
+++
title = "Post Title"
date = "YYYY-MM-DD"
categories = ["category"]
tags = ["tag1", "tag2"]
+++
```
