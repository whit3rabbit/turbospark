import Foundation
import TurboSpark

extension AppModel {
  func makeWorkflowSubagentDispatcher(
    session: TurboSparkSession?,
    project: AppProject?,
    chatID: UUID?
  ) -> WorkflowAppSubagentDispatcher {
    WorkflowAppSubagentDispatcher { agent, prompt, priorHistory, depth in
      // Keep workflow asks on the ordinary runner so filtering, authorization, and depth checks stay active.
      await SubagentRunner.run(
        agent: agent,
        taskPrompt: prompt,
        session: session,
        project: project,
        chatID: chatID,
        depth: depth,
        priorHistory: priorHistory)
    }
  }
}
