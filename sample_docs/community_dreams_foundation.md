# Community Dreams Foundation — Company Handbook (2025 Edition)

## About Us

Community Dreams Foundation (CDF) is a 501(c)(3) nonprofit technology organization founded in 2019, headquartered in Austin, Texas. Our mission is to democratize access to AI-powered productivity tools for underserved communities, nonprofits, and small businesses.

As of Q4 2024, CDF operates across 14 countries, serving 2,300+ nonprofit partners and 47,000 individual users. Annual operating budget for FY2024 was $8.2 million, with 73% funded through grants and 27% through corporate partnerships.

## Leadership Team

- **CEO**: Priya Ramachandran — Former VP of Engineering at Salesforce, joined CDF in 2021.
- **CTO**: Marcus Chen — Built the original TaskMaster platform, holds 3 patents in workflow automation.
- **VP of Product**: Ananya Gupta — Leads the AI-first product strategy, previously at Google DeepMind.
- **VP of Finance**: David Okonkwo — Oversees a $8.2M annual budget, CPA with 15 years nonprofit experience.
- **Head of Partnerships**: Sofia Martinez — Manages 42 corporate partners including Microsoft, AWS, and NVIDIA.

## Organizational Structure

### Engineering Department (38 engineers)
- **Platform Team** (12 engineers): Maintains TaskMaster core platform, handles 2.1M API requests/day.
- **AI/ML Team** (9 engineers): Develops AI assistants, RAG pipelines, and recommendation systems.
- **Infrastructure Team** (8 engineers): Cloud infrastructure on AWS, 99.97% uptime SLA.
- **QA & Security** (5 engineers): Automated testing, SOC 2 Type II compliance, penetration testing.
- **Developer Experience** (4 engineers): Internal tools, CI/CD pipelines, developer onboarding.

### Product Department (14 staff)
- **Product Management** (5 PMs): Each PM owns one product vertical.
- **UX Design** (4 designers): Design system with 120+ components, accessibility-first approach.
- **Data Analytics** (3 analysts): Product metrics, A/B testing framework, user behavior analysis.
- **Technical Writing** (2 writers): API documentation, user guides, changelog management.

### Finance & Operations (11 staff)
- **Project Finance** (4 analysts): Grant tracking, budget allocation, financial modeling.
- **Operations** (4 coordinators): Vendor management, facilities, compliance.
- **HR** (3 specialists): Hiring, culture programs, DEI initiatives.

### Partnerships & Community (8 staff)
- **Corporate Partnerships** (3 managers): Revenue from Microsoft ($1.2M/yr), AWS ($800K/yr), NVIDIA ($450K/yr).
- **Community Programs** (3 coordinators): Hackathons, mentorship, open-source contributions.
- **Marketing & Communications** (2 specialists): Social media, newsletter (28,000 subscribers), blog.

## Key Products

### TaskMaster Pro (Flagship)
An AI-powered project management platform designed for nonprofits. Key metrics:
- **Monthly Active Users**: 47,000 (up 34% YoY)
- **Tasks Completed**: 1.8 million/month
- **Average Response Time**: 142ms (p95: 380ms)
- **Customer Satisfaction (CSAT)**: 4.6/5.0 (from 12,400 survey responses)
- **Churn Rate**: 3.2% monthly (industry average: 5.8%)

Features include:
- Natural language task creation with AI categorization
- Automated workflow templates for 50+ nonprofit verticals
- Grant milestone tracking with compliance dashboards
- Multi-language support (English, Spanish, French, Hindi, Mandarin)
- Integration with Slack, Microsoft Teams, Google Workspace, and Salesforce

### DataBridge (Data Platform)
A unified data platform for nonprofits to consolidate donor databases, program metrics, and financial reporting.
- **Organizations Using**: 340
- **Records Managed**: 12.3 million donor/beneficiary records
- **Data Accuracy Score**: 96.8% (validated via automated reconciliation)
- **Average Query Time**: 230ms for reports spanning 5+ years of data

### AI Assistant (Beta)
A conversational AI assistant that helps nonprofit staff with:
- Document drafting (grant proposals, board reports, impact assessments)
- Data analysis and visualization from DataBridge
- Meeting summarization and action item extraction
- Email drafting with tone and compliance checking

Beta metrics (Q3 2024):
- **Beta Users**: 2,100
- **Queries/Day**: 8,400 average
- **User Satisfaction**: 4.3/5.0
- **Hallucination Rate**: 2.1% (target: <1.5% for GA release)

## Financial Overview (FY2024)

### Revenue Breakdown
| Source | Amount | Percentage |
|--------|--------|-----------|
| Government Grants | $3.1M | 37.8% |
| Foundation Grants | $2.9M | 35.4% |
| Corporate Partnerships | $2.2M | 26.8% |
| **Total** | **$8.2M** | **100%** |

### Expense Breakdown
| Category | Amount | Percentage |
|----------|--------|-----------|
| Engineering & Product | $3.8M | 46.3% |
| Operations & Admin | $1.6M | 19.5% |
| Community Programs | $1.4M | 17.1% |
| Infrastructure (AWS/Cloud) | $0.9M | 11.0% |
| Marketing & Partnerships | $0.5M | 6.1% |
| **Total** | **$8.2M** | **100%** |

### Key Financial Metrics
- **Burn Rate**: $683K/month
- **Runway**: 14 months (with $9.6M committed grants for FY2025)
- **Cost Per User**: $14.50/month (down from $19.20 in FY2023)
- **Grant Renewal Rate**: 89% (industry benchmark: 72%)

## Technology Stack

### Production Infrastructure
- **Cloud**: AWS (us-east-1 primary, eu-west-1 DR)
- **Compute**: EKS (Kubernetes) with 48 nodes, auto-scaling 12-96 pods
- **Database**: PostgreSQL 16 (RDS) + Redis 7 (ElastiCache) + OpenSearch 2.11
- **AI/ML**: NVIDIA A100 GPUs (4x), SageMaker endpoints for model serving
- **CDN**: CloudFront with 23 edge locations
- **Monitoring**: Datadog APM, PagerDuty alerting, 99.97% uptime SLA

### AI/ML Stack
- **LLM**: Llama 3.1 70B (self-hosted) for production, GPT-4 Turbo for evaluation
- **Embeddings**: Cohere embed-v3 (1024-dim) for semantic search
- **Vector DB**: Pinecone (Production), FAISS (Development/Testing)
- **RAG Pipeline**: Custom 4-stage pipeline (retrieve → rerank → filter → generate)
- **Evaluation**: RAGAS framework — Faithfulness: 0.87, Answer Relevancy: 0.91, Context Precision: 0.83

### Security & Compliance
- **SOC 2 Type II** certified (renewed August 2024)
- **GDPR** compliant with data residency controls
- **HIPAA** ready for health-sector nonprofit partners
- **Penetration Testing**: Quarterly via Cobalt, 0 critical findings in last 2 audits
- **Data Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: RBAC with SSO (Okta) and MFA enforcement

## Workflow Processes

### Weekly Cadence
- **Monday**: All-hands standup (30 min), department planning sessions
- **Tuesday–Thursday**: Deep work blocks (no meetings 10am–2pm)
- **Friday**: Demo day (teams showcase weekly progress), retrospectives

### Project Finance Workflow
1. Grant application submitted by Partnerships team
2. Finance team creates budget allocation model (within 5 business days)
3. PM assigned from Product team to track milestones
4. Monthly financial reconciliation against grant terms
5. Quarterly impact reports generated from DataBridge metrics
6. Annual audit coordination with external auditors (Deloitte)

### Engineering Workflow
1. Feature requests triaged weekly by Product team (scored by impact × effort)
2. Engineering specs written with architecture decision records (ADRs)
3. Development follows trunk-based development with feature flags
4. Code review required by 2 engineers, automated via GitHub Actions
5. Staging deployment every Tuesday, Production every Thursday
6. Incident response: P1 < 15 min acknowledgment, P2 < 1 hour

### Recurring Bottleneck
The Asset Management team frequently interfaces with Project Finance for grant reconciliation. Current manual process takes 3-5 business days per grant cycle. An automated reconciliation pipeline has been proposed for Q1 2025, expected to reduce this to <4 hours.

## Q1 2025 Strategic Priorities

1. **AI Assistant GA Launch**: Reduce hallucination rate from 2.1% to <1.5%, add citation support, launch to all 47K users
2. **DataBridge v2**: Real-time data streaming, predictive analytics dashboard, 50% faster queries
3. **International Expansion**: Launch localized versions in Brazil (Portuguese), Japan (Japanese), and Germany (German)
4. **Security Hardening**: Achieve FedRAMP Moderate authorization for government nonprofit partners
5. **Cost Optimization**: Target 20% reduction in cloud infrastructure costs via reserved instances and spot fleet

### Success Metrics for Q1 2025
- AI Assistant: 10,000 daily active users, <1.5% hallucination rate, 4.5+ CSAT
- DataBridge v2: 500 organizations onboarded, <150ms p95 query time
- Revenue: $2.4M in Q1 (from $9.6M annual committed)
- Engineering: 95% sprint completion rate, <2% escaped defect rate
- Infrastructure: Maintain 99.97% uptime, reduce AWS bill by $15K/month

## Contact & Office Information

- **HQ Address**: 401 Congress Avenue, Suite 2100, Austin, TX 78701
- **Engineering Hub**: 155 5th Street, San Francisco, CA 94103
- **Support Email**: support@communitydreams.org
- **General Inquiries**: info@communitydreams.org
- **Engineering Blog**: https://engineering.communitydreams.org
- **Status Page**: https://status.communitydreams.org
