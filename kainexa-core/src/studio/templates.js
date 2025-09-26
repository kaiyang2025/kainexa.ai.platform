// src/studio/templates.js
export const workflowTemplates = {
  customerService: {
    name: "Customer Service Basic",
    description: "기본 고객 서비스 처리 플로우",
    yaml: `
name: Customer Service Basic
version: 1.0
graph:
  - step: classify_intent
    type: intent_classify
    next: route_by_intent
    
  - step: route_by_intent
    type: condition
    params:
      condition: "intent in ['refund', 'exchange', 'cancel']"
      true_branch: handle_order
      false_branch: general_inquiry
      
  - step: handle_order
    type: parallel
    params:
      branches:
        - verify_order
        - check_policy
    next: generate_response
    
  - step: general_inquiry
    type: llm_generate
    params:
      template: "일반 문의 답변 템플릿"
`
  },
  
  technicalSupport: {
    name: "Technical Support",
    description: "기술 지원 에스컬레이션 플로우",
    yaml: `...`
  },
  
  salesInquiry: {
    name: "Sales Inquiry",
    description: "판매 문의 및 상담 플로우",
    yaml: `...`
  }
};