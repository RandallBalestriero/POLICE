# Contributing 

In the context of this project, we would appreciate pull requests to speed-up and improve the proposed POLICE method.
If you find a bug, or would like to suggest an improvement, please open an issue.


### How to make a clean pull request

Look for a project's contribution instructions. If there are any, follow them.
For all below commands be sure to use the ssh version of the urls if needed.

- **Create a personal fork** of the project ([GitHub doc](https://docs.github.com/en/get-started/quickstart/fork-a-repo)) and **Clone the fork** on your local machine
    ```console
    $ git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git # (being in the cloned repo)
    $ cd YOUR_FORK
    $ git remote -v # (being in the cloned repo)
    > origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
    > origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
    ```
    cloning automatically uses `master` branch and `origin` server names
- **Add the original repository as a remote** called `upstream` (for example, use `git remote rename upstream new_name` to rename)
    ```console
    git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git # (being in the cloned repo)
    ```
    Remote repositories are versions of your project that are hosted remotely, collaborating with others involves managing these remote repositories and pushing and pulling data to and from them when you need to share work (to remove one, use `git remote remove paul`). To verify the changes use
    ```console
    $ git remote -v
    > origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
    > origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
    > upstream  https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (fetch)
    > upstream  https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (push)
    ```

- **Create a new branch** to work on! Branch from `develop` if it exists, else from `master`. To create a new branch off main, use `git branch new_branch`. Once created you can then use `git checkout new_branch` to switch to that branch. Additionally, The `git checkout` command accepts a `-b` to create the new branch and immediately switch to it.
- Implement/fix your feature, comment your code
  - following the code style of the project, including indentation.
  - If the project has tests run them!
  - Write or adapt tests as needed.
  - Add or change the documentation as needed.
  - Squash your commits into a single commit with git's [interactive rebase](https://help.github.com/articles/interactive-rebase). Create a new branch if necessary.
- **Push your branch to your fork** on Github, for example to the `master` one.
  ```console
  $ git checkout master
  $ git merge new_branch
  ```
  and to also delete that branch, run `git branch -d new_branch`
- From your fork open a pull request in the correct branch. Target the project's `develop` branch if there is one, else go for `master`!
- If the maintainer requests further changes just push them to your branch. The PR will be updated automatically.
- Once the pull request is approved and merged you can pull the changes from `upstream` to your local repo and delete

- **To keep your fork synced** with the original repo, see [this doc](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork), in short do
    ```console
    $ git fetch upstream
    $ git checkout main
    $ git merge upstream/main
    ```
    The `git checkout` command lets you navigate between the branches created by git branch. Checking out a branch updates the files in the working directory to match the version stored in that branch, and it tells Git to record all new commits on that branch (not to confuse with `git clone` that fetches code from a remote repository while checkout switches between versions of code already on the local system). Think of it as a way to select which line of development you’re working on. To address conflicts, see [this doc](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts)
your extra branch(es).

And last but not least: Always write your commit messages in the present tense. Your commit message should describe what the commit, when applied, does to the code – not what you did to the code.