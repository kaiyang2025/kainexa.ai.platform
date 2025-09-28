export class KakaoClient {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async sendMessage(userId: string, message: string) {
    console.log('Sending Kakao message:', { userId, message });
    return { success: true };
  }
}
