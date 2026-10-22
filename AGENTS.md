**Use standards written by humans for humans, and cite sources when answering questions.** Core references:
   - [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/opinions/): DrivenData's practical conventions for organized, reproducible projects.
   - [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html): Conventions for readable, consistent Python.
   - [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html): Maintainer guidance for implementing and evaluating machine learning.

## Code comments

- Use clear names and simple code first. Add comments when they explain a reason, assumption, constraint, or non-obvious step.
- Write short, concrete sentences. Use the same names as the code and explain unfamiliar domain terms.
- Put block comments immediately above the relevant code. Start each line with `# ` and wrap prose at about 72 characters.
- Use inline comments sparingly. Separate them from code with at least two spaces.
- Keep comments accurate when code changes. Remove comments that repeat obvious operations, stale claims, and commented-out code.
- Use docstrings to explain how to call reusable functions, including inputs, outputs, and relevant side effects.
- In notebooks, use Markdown for objectives, methods, and results. Keep code comments focused on local decisions.

Sources: [Google Python Style Guide, comments and docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) and [PEP 8, comments](https://peps.python.org/pep-0008/#comments). The notebook rule is this project's application of that guidance.
