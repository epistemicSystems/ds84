# Claude Code Configuration

This directory contains configuration files for Claude Code to provide enhanced assistance for the REALTOR AI COPILOT project.

## Files

### claude.md
The main project context file that provides Claude Code with comprehensive understanding of:
- Project architecture and philosophy
- Technology stack
- Domain models
- Workflow architecture
- Development guidelines
- Key considerations and constraints

This file is automatically loaded by Claude Code to provide context-aware assistance.

## Custom Slash Commands

This project includes several custom slash commands to accelerate common development tasks:

### `/setup`
**Description**: Guide me through setting up the REALTOR AI COPILOT development environment

Use this command when you first clone the repository or need to troubleshoot your development environment. Claude will help you:
- Install dependencies
- Configure databases
- Set up environment variables
- Verify the installation

**Example**: Simply type `/setup` and Claude will walk you through the entire process.

---

### `/implement-workflow`
**Description**: Help me implement a new cognitive workflow following the architecture spec

Use this command when you want to add a new workflow (e.g., property search, agent analysis) to the system. Claude will:
- Guide you through the workflow design
- Create necessary prompt templates
- Implement the workflow service
- Create API endpoints
- Write tests

**Example**: Type `/implement-workflow` and Claude will ask which workflow you want to implement.

---

### `/create-prompt`
**Description**: Help me create a new prompt template following best practices

Use this command when you need to create a new prompt template for a cognitive function. Claude will:
- Help you structure the prompt correctly
- Follow meta-prompting patterns
- Create appropriate sections (cognitive_function, context, reasoning, output_format)
- Save it in the right location
- Show you how to use it

**Example**: Type `/create-prompt` and Claude will guide you through the process.

---

### `/test-workflow`
**Description**: Help me test a cognitive workflow implementation

Use this command to test your workflows thoroughly. Claude will:
- Create comprehensive test cases
- Implement unit and integration tests
- Generate example API requests
- Run tests and help debug issues

**Example**: Type `/test-workflow` to start the testing process.

---

### `/optimize-prompt`
**Description**: Help me optimize a prompt template for better performance

Use this command when you want to improve an existing prompt's performance. Claude will:
- Analyze the current prompt
- Generate variant prompts with different optimization strategies
- Test variants against evaluation criteria
- Recommend the best version

**Example**: Type `/optimize-prompt` and specify which prompt to optimize.

---

### `/review-architecture`
**Description**: Review and explain the architectural components of the REALTOR AI COPILOT

Use this command to get a comprehensive explanation of the system architecture. Great for:
- Onboarding new team members
- Understanding how components fit together
- Reviewing design decisions

**Example**: Type `/review-architecture` for a detailed walkthrough.

---

### `/deploy`
**Description**: Help me prepare for deployment of the REALTOR AI COPILOT

Use this command when you're ready to deploy the system. Claude will:
- Review deployment readiness
- Create deployment configurations
- Set up monitoring and logging
- Generate deployment checklists
- Guide you through the deployment process

**Example**: Type `/deploy` when ready for production.

---

## Using Slash Commands

To use a slash command:

1. Type `/` in Claude Code
2. Start typing the command name (e.g., `setup`)
3. Select the command from the autocomplete list
4. Press Enter

Claude will then execute the command and guide you through the process.

## Customizing Commands

To add your own custom commands:

1. Create a new `.md` file in `.claude/commands/`
2. Add frontmatter with the description:
   ```markdown
   ---
   description: Your command description
   ---

   Your prompt instructions...
   ```
3. Save the file and it will be automatically available in Claude Code

## Project Context

The `claude.md` file is automatically loaded by Claude Code to provide project-specific context. This enables Claude to:

- Understand your project architecture
- Follow your coding standards
- Use the correct technology stack
- Apply domain-specific knowledge
- Suggest relevant patterns and practices

You don't need to manually reference this file - Claude Code loads it automatically!

## Best Practices

### When to Use Slash Commands

Use slash commands for:
- ✅ Repetitive tasks (setup, testing, deployment)
- ✅ Complex workflows that require multiple steps
- ✅ Tasks that need to follow specific patterns
- ✅ Onboarding and documentation needs

### When to Ask Directly

Ask Claude directly (without slash commands) for:
- ✅ Quick questions
- ✅ Code reviews
- ✅ Debugging specific issues
- ✅ Custom one-off tasks

## Updating Configuration

### Updating Project Context

Edit `.claude/claude.md` when:
- Architecture changes significantly
- New major features are added
- Technology stack is updated
- Development guidelines change

### Adding New Commands

Create new command files when:
- You have a repetitive multi-step workflow
- You want to codify a common process
- You need to enforce specific patterns
- You want to help onboard new developers

## Resources

- **Main Documentation**: See markdown files in project root
- **Implementation Roadmap**: `IMPLEMENTATION_ROADMAP.md`
- **Quick Start Guide**: `PROJECT_SETUP.md`
- **Claude Code Docs**: https://docs.claude.com/claude-code

---

**Happy Coding! 🚀**
