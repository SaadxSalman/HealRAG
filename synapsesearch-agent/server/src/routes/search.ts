import { Router } from 'express';
import { classifyIntent } from '../agents/intentAgent';

const router = Router();

router.post('/query', async (req, res) => {
  try {
    const { userQuery } = req.body;

    // Phase 1: Intent Analysis
    const classification = await classifyIntent(userQuery);

    // Phase 2: Logic Branching (Placeholder for Retrieval Agent)
    console.log(`User Intent identified as: ${classification.intent}`);
    
    // In a full implementation, you'd call your Retrieval Agent here
    // based on the 'intent' (e.g., searching Weaviate for code vs docs)

    res.json({
      status: "success",
      intent: classification,
      message: "Intent analyzed. Proceeding to retrieval..."
    });
  } catch (error) {
    res.status(500).json({ error: "Agent failed to process query" });
  }
});

export default router;