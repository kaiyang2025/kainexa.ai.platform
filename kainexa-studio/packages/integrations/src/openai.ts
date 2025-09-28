export class OpenAIClient {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async complete(prompt: string) {
    console.log('OpenAI completion:', prompt);
    return { text: 'Mock response' };
  }
}
