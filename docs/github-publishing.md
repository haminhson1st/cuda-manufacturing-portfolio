# Publish the portfolio on GitHub

**Outcome:** one public repository whose status, implementations, and evidence are easy to inspect. Estimated time: 45–60 minutes. This walkthrough uses the connected account name `haminhson1st`; change it if publishing elsewhere.

## 1. Keep the scope focused

Use `cuda-manufacturing-portfolio` as the main repository. The three project directories provide separate case studies. Split a project into its own repository later only when it has an independent build, documentation, and enough original work to stand alone.

The prepared folder contains its own Git repository. Work inside it, not the parent CSE520 KNS directory. The course's textbooks and indexes do not belong in this upload.

## 2. Create an empty remote

Open <https://github.com/new> while signed in. Choose:

- Repository: `cuda-manufacturing-portfolio`
- Description: `CUDA C++ for manufacturing: sensor analytics, visual inspection, and reproducible GPU experiments on ASU Sol.`
- Visibility: Public, if you want recruiters to view it.
- Leave GitHub's automatic README, .gitignore, and license initialization off because this local folder already has content.

If this repository name already exists, inspect it before proceeding. Choose a different name or deliberately integrate with its history; do not overwrite an unrelated repository or force-push.

GitHub documents repository creation in its [repository quickstart](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories).

## 3. Make the initial commit locally

In a PowerShell terminal opened in this portfolio folder:

```powershell
git status --short
git branch --show-current
python scripts/check_docs.py
```

The branch should be `main`. If you copied the files without the `.git` directory, initialize the copied folder with `git init -b main` first.

If Git has no author identity configured, set it for this repository. Replace the placeholders with your preferred identity; a GitHub-provided no-reply address is available in your GitHub email settings.

```powershell
git config user.name "YOUR_DISPLAY_NAME"
git config user.email "YOUR_GITHUB_EMAIL_OR_NOREPLY_ADDRESS"
git add .
git diff --cached --stat
git diff --cached
git commit -m "docs: add CUDA manufacturing learning portfolio"
```

Review the staged files before committing. They should contain the course material and starter code, with no credentials, textbook PDFs, datasets, or fabricated measurements. Decide whether to publish `solutions/`: it holds AI-prepared reference code, so excluding it keeps the public history limited to your own implementations (see [the solutions note](../solutions/README.md)).

## 4. Connect and push

```powershell
git remote -v
git remote add origin https://github.com/haminhson1st/cuda-manufacturing-portfolio.git
git push -u origin main
```

Run `git remote add` only when no `origin` exists. If it already exists, verify its destination before pushing. Authenticate with the credential manager or your normal GitHub authentication method. Do not paste a token into a repository file or remote URL.

## 5. Verify the published page

Open the repository and check the rendered README and tutorial links. In Actions, inspect the first `docs-and-cpu` run. A green CPU workflow does not establish that GPU tests passed on Sol; keep the separate GPU validation record.

Suggested repository topics: `cuda`, `cpp`, `gpu-computing`, `manufacturing`, `computer-vision`, `signal-processing`, `slurm`.

Pin the repository on your profile once it contains a completed lab with evidence. Use an actual output image or measurement plot as the preview when you have one.

## 6. Work in small, explainable changes

At the start of a lab, open a GitHub issue describing its acceptance criteria. Then create a branch:

```bash
git switch -c lab/02-sensor-calibration
# Implement, test, and write your notebook.
git add src tutorials results docs
git diff --cached
git commit -m "feat: calibrate multiple sensor channels"
git push -u origin lab/02-sensor-calibration
```

Open a pull request and use the included template. Review your diff before merging. Update project status only when the linked evidence supports it. Commit actual milestones when they happen; do not backdate activity or split meaningless edits to inflate the graph.

## 7. Publish a case study

After weeks 4, 7, and 12, add a short report based on [the case-study template](../templates/case-study.md). Add a release only when it corresponds to a reproducible milestone. A useful release contains the source commit, build commands, data selection, results manifest, and known limitations.

The strongest portfolio claim is specific and defensible: a problem, your implementation decision, a measured result, and its limits. See [the portfolio playbook](portfolio-playbook.md).
