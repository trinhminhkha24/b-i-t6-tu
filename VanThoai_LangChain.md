# Văn thoại mẫu — Seminar LangChain 1.x
## Phân công: A · B · C · D  |  Thời lượng ước tính: 20–25 phút

---

## 🎤 NGƯỜI A — Mở đầu + §1 Document + §2 Text Splitter  *(~6 phút)*

### [Slide 1 — Title]
> **A:** Xin chào thầy cô và các bạn. Hôm nay nhóm mình sẽ trình bày về **LangChain 1.x** — framework Python để xây dựng ứng dụng LLM — thông qua demo thực tế từng method với nhiều bộ tham số khác nhau.
>
> Nhóm mình gồm 4 người. Mình là A, sẽ phụ trách phần mở đầu, Document và Text Splitter. B sẽ nói về Embeddings và Vector Store. C trình bày Retriever, Prompt Template và Chat Model. Cuối cùng D sẽ đưa mọi thứ lại với nhau qua Output Parser, LCEL, RAG pipeline hoàn chỉnh và Agent.

---

### [Slide 2 — Agenda]
> **A:** Đây là bảng phân công cụ thể. Các bạn có thể thấy toàn bộ pipeline RAG chia thành 11 section, mỗi người phụ trách khoảng 2–3 section. Mình bắt đầu nhé.

---

### [Slide 3 — LangChain là gì?]
> **A:** Trước tiên — LangChain là gì?
>
> LangChain là framework Python mã nguồn mở, cho phép mình ghép nối các thành phần như Document, Text Splitter, Embeddings, Vector Store, Prompt, LLM thành một pipeline hoàn chỉnh.
>
> Điểm mạnh nhất là nó **trừu tượng hoá provider** — dù dùng Google Gemini, OpenAI hay Anthropic, code hầu như giống nhau, chỉ đổi dòng khởi tạo.
>
> Phiên bản 1.x giới thiệu **LCEL** — LangChain Expression Language — cho phép ghép component bằng toán tử `|` như pipe trong terminal. Mình sẽ thấy cụ thể ở phần sau.
>
> Pipeline tổng quát luôn đi theo hướng này *(chỉ vào mũi tên ở cuối slide)*: từ Document thô → cắt nhỏ → embed → lưu vào store → retrieve → prompt → LLM → câu trả lời.

---

### [Slide 4 — §1 Document]
> **A:** Building block đầu tiên và đơn giản nhất: `Document`.
>
> `Document` chỉ có 3 thuộc tính: `page_content` chứa văn bản, `metadata` là dict tuỳ ý để lưu nguồn, số trang, điều khoản... và `id` tự sinh.
>
> *(chỉ vào code box)* Ví dụ ở đây mình tạo một Document từ quy chế đào tạo, kèm metadata là tên file và số điều.
>
> **Tại sao metadata quan trọng?** Vì về sau mình sẽ dùng nó để **lọc** — chỉ tìm trong quy chế, bỏ qua FAQ — và để **trích dẫn nguồn** trong câu trả lời. Nếu không đặt metadata ngay từ đầu, lúc cần thì phải index lại toàn bộ.

---

### [Slide 5 — §2 Text Splitter]
> **A:** Tiếp theo — tại sao phải cắt văn bản?
>
> Context window của model có giới hạn. Chunk quá to thì nhiễu, model đọc không hết. Chunk quá nhỏ thì mất ngữ cảnh, câu bị cụt ý nghĩa.
>
> LangChain dùng `RecursiveCharacterTextSplitter` — nó thử cắt theo danh sách separator ưu tiên từ trên xuống: trước tiên thử `\n\n` *(ngắt đoạn)*, rồi `\n` *(ngắt dòng)*, rồi khoảng trắng, cuối cùng mới cắt từng ký tự.
>
> **Một điểm dễ hiểu sai về `chunk_overlap`** *(chỉ vào hình minh hoạ)*: với văn bản có `\n\n` rõ ràng, `overlap=0` và `overlap=40` cho kết quả **y hệt nhau** — vì splitter đã cắt theo đoạn, không cần overlap. Overlap chỉ kích hoạt khi phải cắt giữa chừng ký tự — tức là văn bản liên tục không có điểm ngắt tự nhiên.
>
> Giá trị khuyến nghị cho tiếng Việt: `chunk_size=800`, `chunk_overlap=150`.
>
> Mình xin nhường micro cho **B** để tiếp tục phần Embeddings.

---
---

## 🎤 NGƯỜI B — §3 Embeddings + §4 Vector Store  *(~5 phút)*

### [Slide 6 — §3 Embeddings]
> **B:** Cảm ơn A. Sau khi có Document và đã cắt xong, bước tiếp theo là biến văn bản thành vector — đây là **Embeddings**.
>
> Embedding model ánh xạ chuỗi văn bản thành vector số thực, sao cho văn bản **nghĩa gần nhau thì vector gần nhau** trong không gian đó.
>
> Có 2 method: `embed_query` cho một câu hỏi đầu vào — trả về 1 vector. `embed_documents` cho danh sách tài liệu — trả về list vector.
>
> *(chỉ vào phần cosine similarity)* Để đo "gần nhau", mình dùng **cosine similarity**. Ví dụ câu hỏi "Đăng ký nhiều nhất bao nhiêu tín chỉ?" có similarity 0.83 với câu "Mỗi HK đăng ký tối đa 24 tín chỉ", nhưng chỉ 0.54 với "Hôm nay Sài Gòn trời mưa" — model thực sự hiểu nghĩa, không chỉ đếm từ khoá.
>
> Trong demo, mình dùng `GoogleGenerativeAIEmbeddings` của Gemini — trả về vector **3072 chiều**.

---

### [Slide 7 — §4 Vector Store]
> **B:** Có vector rồi, mình cần nơi để lưu và tìm kiếm — đó là **Vector Store**.
>
> `InMemoryVectorStore` lưu trong RAM, đủ để demo. Khi lên production chỉ cần đổi sang Chroma, Qdrant hoặc PGVector — **API y hệt nhau**, không cần sửa code logic.
>
> *(lướt qua các method)* 4 method chính: `add_documents` để index, `similarity_search` để tìm k doc gần nhất, `similarity_search_with_score` để kèm điểm số, và `max_marginal_relevance_search` — gọi tắt là MMR.
>
> **MMR** giải quyết một vấn đề thực tế: `similarity_search` thuần thường trả về k doc **gần giống nhau**, lặp đi lặp lại cùng một ý. MMR cân bằng giữa *liên quan tới câu hỏi* và *khác biệt với những gì đã chọn* — tham số `lambda_mult=0.5` là điểm cân bằng tốt.
>
> **Lưu ý quan trọng**: ý nghĩa của `score` **khác nhau tuỳ provider** — InMemoryVectorStore trả khoảng cách nên số **nhỏ hơn** là tốt hơn, một số store khác lại ngược lại.
>
> Mình nhường lại cho **C**.

---
---

## 🎤 NGƯỜI C — §5 Retriever + §6 Prompt Template + §7 Chat Model  *(~6 phút)*

### [Slide 8 — §5 Retriever]
> **C:** Cảm ơn B. Phần tiếp theo là **Retriever** — bước kết nối Vector Store vào pipeline.
>
> `VectorStore` **không phải `Runnable`**, nên không ghép được vào chain bằng `|`. Method `as_retriever()` bọc nó lại thành `Runnable` để có `invoke()`, `batch()`, `stream()` và quan trọng nhất — ghép chain được.
>
> *(chỉ vào code)* Mình config retriever với `search_type="mmr"` và `search_kwargs`. Có 3 cấu hình phổ biến: similarity đơn giản, MMR cân bằng, và `similarity_score_threshold` để lọc theo ngưỡng.
>
> **Ngưỡng là tuyến phòng thủ chống hallucination**: khi câu hỏi nằm ngoài phạm vi kho tài liệu, retriever nên trả về **rỗng** thay vì lôi về doc không liên quan. Nếu retriever lôi về doc không liên quan mà model không được hướng dẫn, nó sẽ **bịa** câu trả lời.

---

### [Slide 9 — §6 Prompt Template]
> **C:** Tiếp theo — **Prompt Template** giúp mình viết prompt tái sử dụng được, có placeholder `{ten_bien}`.
>
> Nhóm mình minh hoạ 3 mức prompt cho cùng một nhiệm vụ — trả lời dựa trên tài liệu:
>
> **Mức 1** chỉ là `"Trả lời câu hỏi: {cau_hoi}"` — không có ngữ cảnh, model sẽ hallucinate.
>
> **Mức 2** thêm `{context}` — tốt hơn nhưng chưa ràng buộc model phải chỉ dùng context.
>
> **Mức 3** là production-ready: có system prompt, có `{context}`, và quan trọng nhất — có câu *"Nếu thông tin không đủ, trả lời đúng câu: 'Tôi không tìm thấy'"*. Câu này là thứ **giảm hallucination mạnh nhất** trong RAG.
>
> Ngoài ra còn `partial()` để điền trước biến cố định như ngày hôm nay, tên trường — tiện khi deploy.

---

### [Slide 10 — §7 Chat Model]
> **C:** Cuối phần của mình — **Chat Model**.
>
> Sau khi init bằng `init_chat_model()`, có 3 method chính: `invoke` cho câu trả lời hoàn chỉnh, `stream` hiển thị dần từng chunk cho UX tốt hơn, và `batch` xử lý nhiều request song song.
>
> *(chỉ vào benchmark)* Trong demo với `gemini-3.5-flash-lite`: tuần tự 3 câu mất 40.6 giây, batch mất 38.6 giây — nhanh hơn 1.1 lần. Với model nặng hơn và nhiều câu hơn, chênh lệch sẽ rõ hơn nhiều.
>
> Về `temperature`: **0** cho RAG và phân loại — muốn kết quả nhất quán, tái hiện được. **Cao** cho brainstorm, đặt tên, viết quảng cáo.
>
> Mình xin nhường **D** để tổng hợp tất cả lại.

---
---

## 🎤 NGƯỜI D — §8 Output Parser + §9 LCEL + §10 RAG + §11 Agent + Kết  *(~7 phút)*

### [Slide 11 — §8 Output Parser + §9 LCEL]
> **D:** Cảm ơn C. Mình sẽ ghép các thành phần lại với nhau.
>
> Trước tiên, **Output Parser** — thứ nhỏ nhưng hầu như chain nào cũng cần. `StrOutputParser` đơn giản chuyển `AIMessage` thành `str`. Vì bước sau thường cần string chứ không cần object message.
>
> Quan trọng hơn là `with_structured_output(Pydantic)` — ép model trả đúng schema. Ví dụ trong demo, mình ép model đọc một điều khoản quy chế và trả về object `{so_dieu, tieu_de, cac_con_so, tom_tat}`. Điều này dùng **function calling** ở tầng API — model bị ép sinh đúng schema, không phải "xin" nó trả JSON rồi hy vọng.
>
> Tiếp theo — **LCEL**, trái tim của LangChain 1.x. *(chỉ vào code)* Mọi component đều là `Runnable`. `Runnable` nào cũng có `invoke`, `stream`, `batch`, `ainvoke`. Và ghép bằng `|`.
>
> Pattern chuẩn của RAG chain trông như thế này: nhánh `context` đưa câu hỏi qua retriever rồi format_docs, nhánh `cau_hoi` đi thẳng qua `RunnablePassthrough`, hai nhánh gộp lại vào prompt, rồi model, rồi parser.

---

### [Slide 12 — §10 Full RAG]
> **D:** Đây là pipeline hoàn chỉnh khi ghép tất cả lại.
>
> *(chỉ vào flow)* Câu hỏi đi vào → Retriever lấy k doc → `format_docs` định dạng với tên nguồn → Prompt Mức 3 → Gemini → StrOutputParser → câu trả lời kèm nguồn.
>
> Mình chạy với 4 câu hỏi test:
>
> Câu "Sinh viên đăng ký tối đa bao nhiêu tín chỉ?" → model trả lời đúng, **kèm nguồn** `[Nguồn: quy_che - §4]`.
>
> Câu "Phúc khảo mất bao nhiêu tiền?" → đúng và kèm nguồn FAQ.
>
> **Câu bẫy quan trọng nhất**: "Học phí một tín chỉ là bao nhiêu?" — thông tin này **không có** trong kho tài liệu. Một RAG được thiết kế tốt phải trả lời *"Tôi không tìm thấy thông tin này trong quy chế"* — và demo của nhóm mình đã làm được điều đó. Đây là tiêu chí quan trọng nhất khi đánh giá RAG.
>
> Ngoài ra còn test ảnh hưởng của `k`: k=1 trả lời cụt, k=3 đủ ý, k=6 hơi dư context.

---

### [Slide 13 — §11 Agent]
> **D:** Cuối cùng — **Agent** — khác biệt cốt lõi so với chain.
>
> Chain có luồng **cố định**: luôn retrieve rồi generate. Agent cho **model tự quyết định** gọi tool nào, theo thứ tự nào, bao nhiêu lần.
>
> *(chỉ vào demo trace)* Câu hỏi demo: *"Em có 3 môn: 3.7 (3tc), 3.0 (4tc), 2.5 (2tc). GPA của em là bao nhiêu và em có đủ điều kiện làm khóa luận không?"*
>
> Câu hỏi này cần **2 tool khác nhau**: `tinh_gpa` để tính điểm, `tra_cuu_quy_che` để tra điều kiện khóa luận. Agent tự nhận ra và gọi cả hai — **điều một chain cố định không làm được**.
>
> Agent phù hợp khi câu hỏi phức tạp, cần suy luận nhiều bước, hoặc không biết trước cần bao nhiêu tool.

---

### [Slide 14 — Bảng tổng kết]
> **D:** Đây là bảng tổng kết các tham số quan trọng nhất. Giá trị đề xuất này là điểm khởi đầu — thực tế cần điều chỉnh dựa trên bộ test câu hỏi cụ thể của từng dự án.
>
> Đặc biệt `score_threshold` — không có con số đúng sẵn. Cách làm đúng là chạy bộ câu hỏi test, in điểm của câu trong phạm vi và câu ngoài phạm vi, rồi chọn ngưỡng nằm giữa hai cụm.

---

### [Slide 15 — Demo Live / Q&A]
> **D:** Nhóm mình chuẩn bị demo live nếu thầy cô và các bạn muốn xem trực tiếp trên notebook.
>
> *(nếu demo)* Đây là file `LangChain_Demo_TungMethod.ipynb`. Mình sẽ chạy từ Section 1 đến Section 11... *(chạy demo)*

---

### [Slide 16 — Kết luận]
> **D:** Tóm lại, qua seminar hôm nay nhóm mình muốn nhấn mạnh 4 điểm:
>
> Một — **8 building block** tạo nên mọi RAG pipeline: Document → Splitter → Embed → Store → Retrieve → Prompt → LLM → Parse.
>
> Hai — **LCEL là trái tim**: ghép bằng `|`, mọi component đều là `Runnable`.
>
> Ba — **Chống hallucination** không phải may mắn: phải dùng Prompt Mức 3, đặt ngưỡng retriever, và luôn test bằng câu bẫy ngoài phạm vi.
>
> Bốn — **Agent mạnh hơn chain** khi câu hỏi phức tạp, nhưng kiểm soát ít hơn — cần cân nhắc theo use case.
>
> Nhóm mình xin dừng ở đây. Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm mình sẵn sàng nhận câu hỏi!

---

## 📋 Ghi chú thêm cho nhóm

| Người | Slide | Thời lượng |
|-------|-------|-----------|
| A | 1 → 5 | ~6 phút |
| B | 6 → 7 | ~5 phút |
| C | 8 → 10 | ~6 phút |
| D | 11 → 16 | ~7 phút |
| **Tổng** | | **~24 phút** |

- **Chuyển micro**: người trước nói "Mình nhường micro cho [tên]" trước khi chuyển slide
- **Demo live** ở Slide 15: chuẩn bị sẵn notebook đã run, API key đặt trong biến môi trường
- **Câu hỏi thường gặp**: "Khác gì LlamaIndex?" → LangChain mạnh về chain/agent linh hoạt, LlamaIndex mạnh về indexing và query phức tạp
