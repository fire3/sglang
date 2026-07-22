import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { defaultTheme } from "@vuepress/theme-default";
import { searchPlugin } from "@vuepress/plugin-search";

export default defineUserConfig({
  lang: "zh-CN",
  title: "SGLang 源码分析",
  description: "逐步深入 SGLang 高性能 LLM 推理框架源码",

  bundler: viteBundler(),

  theme: defaultTheme({
    logo: null,
    repo: "https://github.com/sgl-project/sglang",
    sidebar: [
      {
        text: "总览",
        link: "/",
      },
      {
        text: "阶段一：项目概览与基础架构",
        link: "/01-overview/",
        children: [
          "/01-overview/01-project-intro.md",
          "/01-overview/02-directory-structure.md",
          "/01-overview/03-package-entry.md",
          "/01-overview/04-build-system.md",
          "/01-overview/05-frontend-language.md",
          "/01-overview/06-cli-and-tools.md",
        ],
      },
      {
        text: "阶段二：SRT 核心",
        link: "/02-srt-core/",
        children: [
          "/02-srt-core/01-server-startup.md",
          "/02-srt-core/02-engine-core.md",
          "/02-srt-core/03-api-entrypoints.md",
          "/02-srt-core/04-tokenizer-manager.md",
          "/02-srt-core/05-scheduler.md",
          "/02-srt-core/06-scheduler-components.md",
          "/02-srt-core/07-tp-worker.md",
          "/02-srt-core/08-detokenizer.md",
          "/02-srt-core/09-runtime-context.md",
          "/02-srt-core/10-observability.md",
        ],
      },
    ],
  }),

  plugins: [searchPlugin({})],
});
