import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

class LabelTool:
    def __init__(self, master):
        self.master = master
        self.master.title("Ferramenta de Rotulação de Painéis Solares")
        
        self.brush_size = 10
        self.current_image = None
        self.current_mask = None
        self.draw = None
        self.image_path = ""
        self.mask_path = ""
        
        # Frame de Botões
        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_open = tk.Button(self.btn_frame, text="Abrir Imagem", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_save = tk.Button(self.btn_frame, text="Salvar Máscara", command=self.save_mask)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_clear = tk.Button(self.btn_frame, text="Limpar", command=self.clear_mask)
        self.btn_clear.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Slider de Pincel
        self.brush_scale = tk.Scale(self.btn_frame, from_=2, to=50, orient=tk.HORIZONTAL, label="Tamanho Pincel")
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.LEFT, padx=10)
        
        # Area de desenho (Canvas)
        self.canvas = tk.Canvas(self.master, width=512, height=512, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Eventos do mouse
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        self.photo = None

    def open_image(self):
        # Tenta abrir na pasta train/images por padrão
        initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train", "images"))
        if not os.path.exists(initial_dir):
            initial_dir = os.path.dirname(__file__)
            
        filepath = filedialog.askopenfilename(initialdir=initial_dir, title="Selecione um tile", filetypes=[("JPEG", "*.jpeg *.jpg"), ("PNG", "*.png")])
        if not filepath:
            return
            
        self.image_path = filepath
        self.current_image = Image.open(filepath).convert("RGBA")
        
        # Cria máscara preta
        self.current_mask = Image.new("L", self.current_image.size, 0)
        self.draw = ImageDraw.Draw(self.current_mask)
        
        self.update_canvas()
        
    def update_canvas(self):
        if self.current_image is None: return
        
        # Mescla a imagem original com a máscara para visualização (sobreposição vermelha semitransparente)
        overlay = Image.new("RGBA", self.current_image.size, (255, 0, 0, 0))
        # Usa a máscara para definir a opacidade do vermelho onde pintamos
        overlay.putalpha(self.current_mask.point(lambda p: 128 if p > 0 else 0))
        
        display_img = Image.alpha_composite(self.current_image, overlay)
        self.photo = ImageTk.PhotoImage(display_img)
        
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def paint(self, event):
        if self.current_mask is None: return
        
        x, y = event.x, event.y
        r = self.brush_scale.get()
        
        # Desenha um círculo branco (255) na máscara
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        
        # Redesenha a tela
        self.update_canvas()
        
    def clear_mask(self):
        if self.current_mask is None: return
        self.current_mask = Image.new("L", self.current_image.size, 0)
        self.draw = ImageDraw.Draw(self.current_mask)
        self.update_canvas()

    def save_mask(self):
        if self.current_mask is None: return
        
        # Salva na pasta masks (irmã da pasta images)
        img_dir = os.path.dirname(self.image_path)
        base_dir = os.path.dirname(img_dir) # Sobe um nível (de train/images para train)
        mask_dir = os.path.join(base_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        filename = os.path.basename(self.image_path)
        # Salva a máscara como PNG binário
        mask_filename = filename.replace(".jpeg", ".png").replace(".jpg", ".png")
        save_path = os.path.join(mask_dir, mask_filename)
        
        self.current_mask.save(save_path)
        messagebox.showinfo("Sucesso", f"Máscara salva em:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelTool(root)
    root.mainloop()
