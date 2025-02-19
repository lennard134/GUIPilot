import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import CrossEncoder
import pandas as pd
import os
import statistics
#Helper functions from the provided code


def document_parsing(file_path, chunk_size):
    """
    Document parser, processes uploaded document and splits text into chunks for a given chunksize.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_size/10, length_function=len)
    return text_splitter.split_documents(pages)

#GUI Implementation
class ChatApp:
    # TODO: Summarize chat history before feeding back into model.
    
    def __init__(self, root:tk.Tk, embedding: OllamaEmbeddings, rag_model_name: str, sherlock_model_name: str, chunk_size: int):
        """ Init, it inits...."""
        self.root = root
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.rag_model_name = rag_model_name
        self.sherlock_model_name = sherlock_model_name
        self.root.title("Prototype")
        self.chat_history = []
        self.chat_summary = None
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        self.mean_relevance = []
        #Model selection and initialization        self.model_name = 'llama3.2:latest'
        # self.rag_model = ChatOllama(model=self.model_name)
        self.rag_model = self.initialize_model(rag_model_name)
        self.sherlock_model = self.initialize_model(sherlock_model_name)
        
        #Chat History Display
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=80)
        self.chat_display.pack(pady=10)

        #File Upload Button
        self.upload_button = tk.Button(root, text="Upload File", command=self.upload_file)
        self.upload_button.pack(pady=10)

    def on_enter_pressed(self, event):
        self.ask_question()

    # Initialize two LLaMA models
    def initialize_model(self, model_name):
        model = ChatOllama(model=model_name)
        return model

    def summarize_history(self):
        """Generate a summary of the chat history."""
        if not self.chat_history:
            messagebox.showinfo("Info", "Chat history is empty. Start a conversation first.")
            return
        
        try:
            #Concatenate chat history into a single string for 
            history_text = "\n".join([f"Question: {q}\n Answer: {a}" for q, a in self.chat_history])
            
            #Use the model to summarize the history
            summary = self.rag_model.invoke(f"Please summarize the following conversation:\n{history_text}")
            
            # self.log_chat("System", f"Chat history summarized.")
            return summary.content

        except Exception as e:
            self.log_chat("System", f"Error summarizing chat history: {e}")
            messagebox.showerror("Error", f"Error summarizing chat history: {e}")
            return None


    def log_chat(self, role, message):
        """Log messages in the chat display."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{role}: {message}\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def debug(self, text):

        self.log_chat('\nLennarDebugger', text)

    def upload_file(self):
        """Handle file upload and initialize vectorstore."""

        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")])
        if file_path:
            self.log_chat("System", f"Processing file: {file_path}")
            try:
                docs = document_parsing(file_path, chunk_size=self.chunk_size)
                local_embeddings = OllamaEmbeddings(model=self.embedding)
                text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size/10)
                texts = text_splitter.split_documents(docs)
                self.vectorstore = Chroma.from_documents(documents=texts, embedding=local_embeddings)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 4})
                self.log_chat("System", "File uploaded and processed successfully.")
            except Exception as e:
                self.log_chat("System", f"Error processing file: {e}")
                messagebox.showerror("Error", f"Could not process the file: {e}")
        
        try:
            cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1)
            for turns in [8]:#[2,4,6,8,10]:
                with open('../Data/content/SherlockPrompt.txt', 'r') as file:
                    sherlock_text = file.read().replace('\n', '')
                sherlock_prompt = f"""
                    {sherlock_text}  
                """
        
                #Initial prompt starting from RAG inviting for questions
                initial_prompt  = """Hello, as you know I have access to different documents: Journal of the victim, 
                Excerpt from church records, newspaper clippigs, scrawled notes, tunnel sketches, and cryptic code message.
                Let us get started now, what would you like to know about these documents?"""
                
                # Log and update chat history
                self.log_chat("Botanic", initial_prompt)
                self.chat_history.append(('Start message message', initial_prompt))

                for _ in range(turns):
                    sherlock_prompt += "\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.chat_history])
                    sherlock_output = self.sherlock_model.invoke(sherlock_prompt)   
                    sherlock_message = sherlock_output.content
                    self.log_chat("Sherlock", sherlock_message)
                    retrieved_docs = self.retriever.get_relevant_documents(sherlock_message)
                    ## Ranking from here - not ranking intial prompt as no relevant questions often asked
                    
                    response = [[sherlock_message, doc_text.page_content] for doc_text in retrieved_docs]
                    scores = cross_encoder.predict(response)

                    scored_docs = list(zip(scores, retrieved_docs))
                    print(scores)
                    self.mean_relevance.append(statistics.mean(scores))
                    filtered_docs = [doc for score, doc in scored_docs if score != 5]
                    print(len(filtered_docs))
                    retrieved_context = "\n\n".join([doc.page_content for doc in filtered_docs])

                    if not retrieved_context.strip() and len(filtered_docs) > 0:
                        bot_response = "I don’t have enough information from the documents to answer that."
                    else:
                        # Generate response from the LLM
                        rag_prompt =f"""You are assisting a human player in solving a murder mystery game, you have access to 
                                        documents the other player does not have.
                                        Answer the following question based solely on the retrieved documents.
                                        Rules:
                                            1. Use only the retrieved context provided. Do not fabricate answers or add external 
                                            information.
                                            2. Encourage the user to explore further if the retrieved context is insufficient or unclear.
                                            3. If you cannot answer based on the retrieved context, say: "I do not have enough 
                                        information from the documents to answer that."
                                        Inputs:
                                            - Retrieved Context: {retrieved_context}
                                            - User Question: {sherlock_message}
                                        Provide a concise and conversational response. If necessary, guide the user toward 
                                        relevant clues or suggest possible areas for further exploration.
                                        The chat history can be found here:
                                        {self.chat_summary}
                        """ 

                        result = self.rag_model.invoke(rag_prompt)
                        bot_response = result.content#['answer']
                        # Log and update chat history
                        self.log_chat("Botanic", bot_response)
                        self.chat_history.append((sherlock_message, bot_response))
                        self.chat_summary = self.summarize_history()
            
                final_prompt = f"""
                    You are given a chat history of a person and an LLM trying to solve a murder mysterie based on different clues.
                    Try to answer the following questions based on this chat history:
                    1. Who killed Van Der Meer?
                    2. Why did this person kill Van Der Meer?
                    based on the chat history:
                    {"\n".join([f" 'question' {q} 'answer' {a}" for q, a in self.chat_history])}
                    If you are not able to answer these questions, please respond with that you are not able to answer.
                """

                user_output = self.sherlock_model.invoke(final_prompt)
                user_message = user_output.content
                self.log_chat("User", user_message)
                self.chat_history.append(('Prediction:', user_message))
                self.chat_summary = self.summarize_history()
                save_path = "../Results/LLMConversations/"
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                # _, _, files = next(os.walk(save_path))
                # idx = len(files) + 1

                save_name = save_path + str(turns) + 'rag' + self.rag_model_name + 'sherlock' + self.sherlock_model_name + '.txt'
                with open(save_name, 'w') as f:
                    for q,a in self.chat_history:
                        f.write(f"{q, a}\n")
                self.chat_history = []
                self.log_chat("Sys", 'Chat reset')
            print(f'Average relevance over prompts: {self.mean_relevance}')
        except Exception as e:
            self.log_chat("System", f"Error answering question: {e}")
            messagebox.showerror("Error", f"Error answering question: {e}")

#Create and run the app
if __name__ == "__main__":
    """Main loop"""
    rag_model_name = "llama3.2:latest"
    sherlock_model_name = "llama3:latest"
    embedding = "nomic-embed-text"
    chunk_size = 512
    root = tk.Tk()
    app = ChatApp(root, embedding, rag_model_name, sherlock_model_name, chunk_size)
    root.mainloop()

#llama3:latest              365c0bd3c000    4.7 GB    params: 8,03B
#llama3.2:1b                baf6a787fdff    1.3 GB    params: 1B
#llama3.2:latest            a80c4f17acd5    2.0 GB    params: 3,21B
#Embedding
#nomic-embed-text:latest    0a109f422b47    274 MB         
"""
Begin the Game:
Start by asking the human about their documents:
	"What do your documents reveal about the crime scene or key individuals involved?"
	"Do any of your documents mention a motive, timeline, or suspicious behavior?"""