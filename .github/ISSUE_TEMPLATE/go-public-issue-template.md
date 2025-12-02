---
name: Go Public Issue Template
about: The repository publication/open sourcing checklist
title: Go Public
labels: ''
assignees: ''

---

- [ ] No sensitive information is contained in the repo or in commits
> Note to developers: having committed any private information to the repo and then removing it
> afterwards, that private information still is contained in the repo. The only way to remove this is
> with a tool, and then overwriting the entire repo. Please notify security@crabnebula.dev details in
> [SECURITY.md](../../SECURITY.md).

- [ ] Branch protection is enabled and no workflows are run on external forks
- [ ] Dependabot is enabled for security alerts
- [ ] Commit signing is enforced
- [ ] Project is OKed from the security team
- [ ] Only used features of the project are enabled (Wikis,PRs, Issues)
- [ ] License is agreed upon by all parties included. In case of disputes, defer to seniority.
- [ ] Asset licenses vs. code licenses have been considered
- [ ] README is clearly written
- [ ] CONTRIBUTING.md exists with clear guidelines to enable community contributions and a code of conduct
- [ ] Extended documentation is available for larger projects
- [ ] CrabNebula branding is in place
- [ ] Repo header image is in the README
- [ ] Content drafted and ready to share
   - [ ]  Blog posts
   - [ ] X/LinkedIn/Social media posts
   - [ ] ProductHunt posts (for products)
